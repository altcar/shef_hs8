/**
 * wizzibiz Worker
 *
 *   GET  /api/deck?session=...     the swipe deck, re-ranked by your taste model
 *   POST /api/swipe                { session, clip, liked } -> records a swipe
 *   GET  /api/taste?session=...    fits logistic regression on your likes and
 *                                  blends the pitch contours of liked clips
 *   POST /api/classify             raw WAV body -> species prediction
 *
 * Static files (UI, clips, clips.json) are served by Workers static assets.
 */
import deckData from '../public/clips.json'
import { DEFAULTS, decodeWav, extractFeatures, FEATURE_DIM } from './dsp.ts'
import { loadModel, predictWindows, type Prediction } from './model.ts'
import { MODEL_DATA, STATS } from './model-data.ts'

export interface Env {
  DB: D1Database
  ASSETS: Fetcher
}

type Clip = (typeof deckData)['clips'][number]

const model = loadModel(MODEL_DATA)
const stats = {
  mean: Float32Array.from(STATS.mean),
  std: Float32Array.from(STATS.std),
}
const clips: Clip[] = deckData.clips
const clipById = new Map(clips.map((c) => [c.id, c]))

const json = (data: unknown, init: ResponseInit = {}) =>
  new Response(JSON.stringify(data), {
    ...init,
    headers: { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store', ...(init.headers ?? {}) },
  })

function embeddingOf(clip: Clip): Float32Array {
  return Float32Array.from(clip.embedding)
}

/** logistic regression: liked (1) vs skipped (0) over the model's embeddings */
function fitTaste(rows: { clip: Clip; liked: number }[], epochs = 400) {
  const dim = rows[0].clip.embedding.length
  const w = new Float32Array(dim)
  let b = 0
  const lr = 0.4
  const l2 = 0.02

  const sigmoid = (z: number) => 1 / (1 + Math.exp(-z))
  const score = (e: Float32Array) => {
    let z = b
    for (let i = 0; i < dim; i++) z += w[i] * e[i]
    return z
  }

  for (let epoch = 0; epoch < epochs; epoch++) {
    const grad = new Float32Array(dim)
    let gradB = 0
    for (const row of rows) {
      const e = embeddingOf(row.clip)
      const err = sigmoid(score(e)) - row.liked
      for (let i = 0; i < dim; i++) grad[i] += err * e[i]
      gradB += err
    }
    const n = rows.length
    for (let i = 0; i < dim; i++) w[i] -= lr * (grad[i] / n + l2 * w[i])
    b -= lr * (gradB / n)
  }

  return { w, b, sigmoid, score }
}

/**
 * Honest accuracy: leave-one-out when there are few swipes, an 80/20 holdout
 * once there are enough. In-sample accuracy on 8 points is meaningless.
 */
function tasteAccuracy(rows: { clip: Clip; liked: number }[]): number {
  const classify = (model: { sigmoid: (z: number) => number; score: (e: Float32Array) => number }, clip: Clip) =>
    model.sigmoid(model.score(embeddingOf(clip))) >= 0.5 ? 1 : 0

  if (rows.length <= 40) {
    let correct = 0
    for (let i = 0; i < rows.length; i++) {
      const rest = rows.filter((_, j) => j !== i)
      if (new Set(rest.map((r) => r.liked)).size < 2) return 0.5
      const m = fitTaste(rest, 220)
      if (classify(m, rows[i].clip) === rows[i].liked) correct++
    }
    return correct / rows.length
  }

  const cut = Math.floor(rows.length * 0.8)
  const train = rows.slice(0, cut)
  const test = rows.slice(cut)
  const m = fitTaste(train)
  const correct = test.filter((r) => classify(m, r.clip) === r.liked).length
  return correct / test.length
}

function tasteScore(w: Float32Array, b: number, clip: Clip): number {
  let z = b
  for (let i = 0; i < w.length; i++) z += w[i] * clip.embedding[i]
  return 1 / (1 + Math.exp(-z))
}

/** average the pitch tracks of the clips you liked into one new contour */
function blendContours(liked: Clip[]) {
  if (!liked.length) return null
  const points = liked[0].contour.length

  // pitch: median across the liked clips at each instant, ignoring silence
  const hz = new Array(points).fill(0)
  for (let i = 0; i < points; i++) {
    const column = liked.map((c) => c.contour[i]).filter((v) => v > 60)
    if (!column.length) continue
    column.sort((a, b) => a - b)
    hz[i] = column[Math.floor(column.length / 2)]
  }
  // fill gaps between syllables so the synthesised call glides instead of clicking
  let last = hz.find((v) => v > 0) ?? 0
  for (let i = 0; i < points; i++) {
    if (hz[i] > 0) last = hz[i]
    else hz[i] = last
  }
  for (let i = points - 2; i >= 0; i--) if (hz[i] === 0) hz[i] = hz[i + 1]

  const env = new Array(points).fill(0)
  for (const clip of liked) for (let i = 0; i < points; i++) env[i] += (clip.envelope[i] ?? 0) / liked.length

  return {
    hz: hz.map((v) => Math.round(v)),
    env: env.map((v) => Math.round(v * 100) / 100),
    medianHz: Math.round([...hz].filter((v) => v > 0).sort((a, b) => a - b)[Math.floor(hz.filter((v) => v > 0).length / 2)] ?? 0),
  }
}

async function deck(request: Request, env: Env) {
  const session = new URL(request.url).searchParams.get('session') ?? ''
  let swipes: { clip: string; liked: number }[] = []
  if (session) {
    const { results } = await env.DB.prepare('SELECT clip, liked FROM swipes WHERE session = ?').bind(session).all<{ clip: string; liked: number }>()
    swipes = results ?? []
  }
  const seen = new Set(swipes.map((s) => s.clip))

  let ranked = clips
  let taste: { accuracy: number } | null = null
  const rows = swipes.map((s) => ({ clip: clipById.get(s.clip)!, liked: s.liked })).filter((r) => r.clip)
  if (rows.length >= 4 && new Set(rows.map((r) => r.liked)).size === 2) {
    const fitted = fitTaste(rows)
    ranked = [...clips].sort((a, b) => tasteScore(fitted.w, fitted.b, b) - tasteScore(fitted.w, fitted.b, a))
    taste = { accuracy: tasteAccuracy(rows) }
  }

  return json({
    model: deckData.model,
    dsp: deckData.dsp,
    counts: { clips: clips.length, swipes: swipes.length, liked: swipes.filter((s) => s.liked).length },
    taste: taste ? { accuracy: Math.round(taste.accuracy * 1000) / 1000, swipes: rows.length } : null,
    clips: ranked
      .filter((c) => !seen.has(c.id))
      .map((c) => ({
        id: c.id,
        file: c.file,
        label: c.label,
        name: c.name,
        sci: c.sci,
        seconds: c.seconds,
        peakHz: c.peakHz,
        contour: c.contour,
        pca: c.pca,
        top: c.top,
      })),
  })
}

async function swipe(request: Request, env: Env) {
  const body = (await request.json()) as { session?: string; clip?: string; liked?: boolean }
  const session = String(body.session ?? '').slice(0, 64)
  const clip = String(body.clip ?? '')
  if (!session || !clipById.has(clip)) return json({ error: 'bad request' }, { status: 400 })

  await env.DB.prepare(
    `INSERT INTO swipes (id, session, clip, liked, created_at) VALUES (?, ?, ?, ?, ?)
     ON CONFLICT(id) DO UPDATE SET liked = excluded.liked, created_at = excluded.created_at`,
  )
    .bind(`${session}:${clip}`, session, clip, body.liked ? 1 : 0, Date.now())
    .run()

  return json({ ok: true })
}

async function taste(request: Request, env: Env) {
  const session = new URL(request.url).searchParams.get('session') ?? ''
  if (!session) return json({ error: 'missing session' }, { status: 400 })

  const { results } = await env.DB.prepare('SELECT clip, liked FROM swipes WHERE session = ? ORDER BY created_at DESC')
    .bind(session)
    .all<{ clip: string; liked: number }>()

  const rows = (results ?? []).map((r) => ({ clip: clipById.get(r.clip)!, liked: r.liked })).filter((r) => r.clip)
  const liked = rows.filter((r) => r.liked === 1).map((r) => r.clip)

  if (rows.length < 4 || liked.length === 0 || liked.length === rows.length) {
    return json({ ready: false, swipes: rows.length, liked: liked.length, needed: 4 })
  }

  const model = fitTaste(rows)
  const speciesCount = new Map<string, number>()
  for (const clip of liked) speciesCount.set(clip.name, (speciesCount.get(clip.name) ?? 0) + 1)

  const contour = blendContours(liked)

  return json({
    ready: true,
    swipes: rows.length,
    liked: liked.length,
    accuracy: Math.round(tasteAccuracy(rows) * 1000) / 1000,
    likingRate: Math.round((liked.length / rows.length) * 100) / 100,
    topSpecies: [...speciesCount.entries()].sort((a, b) => b[1] - a[1]).slice(0, 5).map(([name, count]) => ({ name, count })),
    contour,
  })
}

async function classify(request: Request) {
  const body = await request.arrayBuffer()
  if (body.byteLength < 44) return json({ error: 'send a WAV file' }, { status: 400 })
  if (body.byteLength > 8 * 1024 * 1024) return json({ error: 'file too large (8 MB max)' }, { status: 413 })

  let prediction: Prediction
  try {
    const audio = decodeWav(new Uint8Array(body))
    const features = extractFeatures(audio, stats.mean, stats.std)
    prediction = predictWindows(model, features)
  } catch (error) {
    return json({ error: `could not read that audio: ${(error as Error).message}` }, { status: 400 })
  }

  return json({
    top: prediction.top.map((t) => ({ code: model.classes[t.index], p: Math.round(t.p * 1000) / 1000 })),
    notABird: Math.round((prediction.probs[model.classes.indexOf('not_a_bird')] ?? 0) * 1000) / 1000,
    dim: FEATURE_DIM,
    sampleRate: DEFAULTS.sampleRate,
  })
}

export default {
  async fetch(request: Request, env: Env, _ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url)
    try {
      if (url.pathname === '/api/deck') return await deck(request, env)
      if (url.pathname === '/api/swipe' && request.method === 'POST') return await swipe(request, env)
      if (url.pathname === '/api/taste') return await taste(request, env)
      if (url.pathname === '/api/classify' && request.method === 'POST') return await classify(request)
      if (url.pathname.startsWith('/api/')) return json({ error: 'not found' }, { status: 404 })
    } catch (error) {
      console.error('api error', url.pathname, (error as Error).message)
      return json({ error: 'internal error' }, { status: 500 })
    }
    return env.ASSETS.fetch(request)
  },
} satisfies ExportedHandler<Env>
