/**
 * export.ts
 *
 * 1. scores every clip with the trained CNN
 * 2. picks the demo deck: most confident clips per species (+ a few negatives)
 * 3. writes those clips as 16 kHz mono WAVs into public/clips/
 * 4. writes public/clips.json with predictions, embeddings and pitch contours
 * 5. quantises the model to int8 and writes src/model-data.ts for the Worker
 */
import { asyncBufferFromFile, parquetReadObjects } from 'hyparquet'
import { mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs'
import { DEFAULTS, FEATURE_DIM, decodeWav, encodeWav16, logMel, normaliseEnvelope, pitchTrack, resample } from '../src/dsp.ts'
import { forward, type Model, type QuantisedLayer, type QuantisedModel, type Spec } from '../src/model.ts'
import { LABELS, LABEL_INDEX } from '../src/species.ts'

const CACHE = 'C:/Users/c85-lau/AppData/Local/Temp/opencode/birds/wizzibiz/cache'
const ROOT = 'C:/Users/c85-lau/AppData/Local/Temp/opencode/birds/wizzibiz'
const DATA_DIR = 'C:/Users/c85-lau/AppData/Local/Temp/opencode/birds/data'
const PER_SPECIES = 4
const NEGATIVES = 8
const WINDOWS = 3

const SHARDS = [
  { path: `${DATA_DIR}/birds-00000.parquet`, source: 'bird', key: 'b0', rows: 1600 },
  { path: `${DATA_DIR}/birds-00001.parquet`, source: 'bird', key: 'b1', rows: 1599 },
  { path: `${DATA_DIR}/esc50-00000.parquet`, source: 'esc50', key: 'e', rows: 984 },
]

/** Clip ids must be unique across shards, so they carry the shard key. */
function shardOf(index: number): number {
  return index < SHARDS[0].rows ? 0 : index < SHARDS[0].rows + SHARDS[1].rows ? 1 : 2
}
function idFor(index: number, row: number): string {
  return `${SHARDS[shardOf(index)].key}-${row}`
}

// ---------------------------------------------------------------- model -----
const trained = JSON.parse(readFileSync(`${CACHE}/model.json`, 'utf8')) as {
  inputH: number
  inputW: number
  spec: Spec[]
  classes: string[]
  weights: ({ w: number[]; b: number[] } | null)[]
  valAccuracy: number
  perClass: { code: string; name: string; total: number; recall: number }[]
}
const stats = JSON.parse(readFileSync(`${CACHE}/stats.json`, 'utf8')) as { mean: number[]; std: number[] }
const meta = JSON.parse(readFileSync(`${CACHE}/meta.json`, 'utf8')) as {
  windows: number
  clips: { id: string; source: string; row: number; label: string; seconds: number; rms: number; peakHz: number; contour: number[]; envelope: number[] }[]
}
const raw = readFileSync(`${CACHE}/raw.f32`)
const feats = new Float32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength))

const model: Model = {
  inputH: trained.inputH,
  inputW: trained.inputW,
  spec: trained.spec,
  classes: trained.classes,
  weights: trained.weights.map((w) => (w ? { w: Float32Array.from(w.w), b: Float32Array.from(w.b) } : null)),
}

function standardised(clipIndex: number, w: number): Float32Array {
  const base = clipIndex * WINDOWS * FEATURE_DIM + w * FEATURE_DIM
  const out = new Float32Array(FEATURE_DIM)
  for (let i = 0; i < FEATURE_DIM; i++) {
    const band = Math.floor(i / DEFAULTS.poolFrames)
    out[i] = (feats[base + i] - stats.mean[band]) / (stats.std[band] || 1)
  }
  return out
}

// -------------------------------------------------------- score all clips ---
type Scored = {
  index: number
  id: string
  label: string
  conf: number
  top: { index: number; p: number }[]
  embedding: Float32Array
  spectrum: Float32Array
  pca: [number, number]
}

console.log(`scoring ${meta.clips.length} clips...`)
const scored: Scored[] = meta.clips.map((clip, index) => {
  const sum = new Array(trained.classes.length).fill(0)
  let embedding = new Float32Array(0)
  const spectrum = new Float32Array(DEFAULTS.nMels)

  for (let w = 0; w < WINDOWS; w++) {
    const x = standardised(index, w)
    const { logits, embedding: e } = forward(model, x)
    embedding = e
    let max = -Infinity
    for (const v of logits) max = Math.max(max, v)
    let total = 0
    const probs = new Float32Array(logits.length)
    for (let c = 0; c < logits.length; c++) {
      probs[c] = Math.exp(logits[c] - max)
      total += probs[c]
    }
    for (let c = 0; c < logits.length; c++) sum[c] += probs[c] / total / WINDOWS
    for (let m = 0; m < DEFAULTS.nMels; m++) {
      let acc = 0
      for (let t = 0; t < DEFAULTS.poolFrames; t++) acc += x[m * DEFAULTS.poolFrames + t]
      spectrum[m] += acc / DEFAULTS.poolFrames / WINDOWS
    }
  }

  const top = sum.map((p, index) => ({ index, p })).sort((a, b) => b.p - a.p).slice(0, 3)
  const trueIndex = LABEL_INDEX.get(clip.label) ?? LABELS.length - 1
  return { index, id: idFor(index, clip.row), label: clip.label, conf: sum[trueIndex], top, embedding, spectrum, pca: [0, 0] }
})

// ------------------------------------------------------------- PCA (2-D) ----
const bands = DEFAULTS.nMels
const mu = new Float32Array(bands)
for (const s of scored) for (let m = 0; m < bands; m++) mu[m] += s.spectrum[m] / scored.length

const cov = new Float32Array(bands * bands)
for (const s of scored) {
  for (let a = 0; a < bands; a++) {
    const da = s.spectrum[a] - mu[a]
    for (let b = 0; b < bands; b++) cov[a * bands + b] += (da * (s.spectrum[b] - mu[b])) / scored.length
  }
}

function principal(matrix: Float32Array, size: number, seedVec: number): Float32Array {
  let vec = new Float32Array(size)
  for (let i = 0; i < size; i++) vec[i] = Math.sin(i * 0.7 + seedVec) + 1e-3
  for (let iter = 0; iter < 80; iter++) {
    const next = new Float32Array(size)
    for (let a = 0; a < size; a++) {
      let acc = 0
      for (let b = 0; b < size; b++) acc += matrix[a * size + b] * vec[b]
      next[a] = acc
    }
    let norm = 0
    for (const v of next) norm += v * v
    norm = Math.sqrt(norm) || 1
    for (let i = 0; i < size; i++) vec[i] = next[i] / norm
  }
  return vec
}

const pc1 = principal(cov, bands, 1)
let lambda = 0
for (let a = 0; a < bands; a++) for (let b = 0; b < bands; b++) lambda += pc1[a] * cov[a * bands + b] * pc1[b]
const deflated = Float32Array.from(cov)
for (let a = 0; a < bands; a++) for (let b = 0; b < bands; b++) deflated[a * bands + b] -= pc1[a] * pc1[b] * lambda * 1.02
const pc2 = principal(deflated, bands, 7)
console.log(`pca: pc1 explains ${lambda.toFixed(1)} variance`)

for (const s of scored) {
  let x = 0
  let y = 0
  for (let m = 0; m < bands; m++) {
    const d = s.spectrum[m] - mu[m]
    x += d * pc1[m]
    y += d * pc2[m]
  }
  s.pca = [Math.round(x * 100) / 100, Math.round(y * 100) / 100]
}

// ------------------------------------------------------ pick the demo deck ---
const bySpecies = new Map<string, Scored[]>()
for (const s of scored) {
  if (s.label === 'not_a_bird') continue
  const list = bySpecies.get(s.label) ?? []
  list.push(s)
  bySpecies.set(s.label, list)
}

const deck: Scored[] = []
for (const list of bySpecies.values()) {
  list.sort((a, b) => b.conf - a.conf)
  deck.push(...list.slice(0, PER_SPECIES))
}
const negatives = scored
  .filter((s) => s.label === 'not_a_bird')
  .sort((a, b) => b.conf - a.conf)
  .slice(0, NEGATIVES)
deck.push(...negatives)

console.log(`deck: ${deck.length} clips (${bySpecies.size} species + ${negatives.length} negatives)`)

// shuffle so consecutive cards are not all the same species
let shuffleSeed = 991
const shuffleRand = () => {
  shuffleSeed = (shuffleSeed * 1103515245 + 12345) & 0x7fffffff
  return shuffleSeed / 0x7fffffff
}
for (let i = deck.length - 1; i > 0; i--) {
  const j = Math.floor(shuffleRand() * (i + 1))
  ;[deck[i], deck[j]] = [deck[j], deck[i]]
}

// ------------------------------------------- pull the audio for the deck ----
mkdirSync(`${ROOT}/public/clips`, { recursive: true })
// clear stale clips from previous exports (ids are shard-scoped now)
for (const file of readdirSync(`${ROOT}/public/clips`)) {
  if (file.endsWith('.wav')) rmSync(`${ROOT}/public/clips/${file}`)
}
const byShard = new Map<string, Scored[]>()
for (const s of deck) {
  const key = SHARDS[shardOf(s.index)].key
  const list = byShard.get(key) ?? []
  list.push(s)
  byShard.set(key, list)
}

const audioLength = new Set<string>()
/** pitch contour + envelope, recomputed from the audio itself */
const contours = new Map<string, { contour: number[]; envelope: number[]; peakHz: number }>()

function downsample(values: Float32Array, points: number): number[] {
  const out: number[] = []
  const step = values.length / points
  for (let i = 0; i < points; i++) {
    const from = Math.floor(i * step)
    const to = Math.min(values.length, Math.max(from + 1, Math.floor((i + 1) * step)))
    let acc = 0
    for (let j = from; j < to; j++) acc += values[j]
    out.push(Math.round((acc / (to - from)) * 10) / 10)
  }
  return out
}

for (const shard of SHARDS) {
  const wanted = byShard.get(shard.key)
  if (!wanted?.length) continue
  const wantedRows = new Map(wanted.map((s) => [meta.clips[s.index].row, s]))

  console.log(`reading ${shard.path} for ${wantedRows.size} clips...`)
  const file = await asyncBufferFromFile(shard.path)
  const rows = await parquetReadObjects({
    file,
    columns: shard.source === 'esc50' ? ['audio', 'category'] : ['audio', 'label'],
    utf8: false,
  })

  for (const [row, scoredClip] of wantedRows) {
    const audio = (rows[row] as { audio?: { bytes?: Uint8Array } })?.audio
    if (!audio?.bytes) continue
    const decoded = decodeWav(audio.bytes)
    const samples = decoded.sampleRate === DEFAULTS.sampleRate ? decoded.samples : resample(decoded.samples, decoded.sampleRate, DEFAULTS.sampleRate)
    writeFileSync(`${ROOT}/public/clips/${scoredClip.id}.wav`, encodeWav16(samples, DEFAULTS.sampleRate))
    audioLength.add(scoredClip.id)

    const mel = logMel(samples, DEFAULTS.sampleRate)
    const track = pitchTrack(mel.mag, mel.frames, mel.bins, DEFAULTS.sampleRate, mel.nFft, mel.hop)
    const contour = downsample(track.hz, 50)
    const pitched = contour.filter((v) => v > 60).sort((a, b) => a - b)
    contours.set(scoredClip.id, {
      contour,
      envelope: downsample(normaliseEnvelope(track.energy), 50).map((v) => Math.round(v * 100) / 100),
      peakHz: Math.round(pitched[Math.floor(pitched.length * 0.9)] ?? 0),
    })
  }
}

// --------------------------------------------------------------- outputs ----
const clipData = deck
  .filter((s) => audioLength.has(s.id))
  .map((s) => {
    const clip = meta.clips[s.index]
    const species = LABELS[LABEL_INDEX.get(clip.label) ?? LABELS.length - 1]
    const pitch = contours.get(s.id)
    return {
      id: s.id,
      file: `clips/${s.id}.wav`,
      label: clip.label,
      name: species.name,
      sci: species.sci,
      seconds: clip.seconds,
      peakHz: pitch?.peakHz ?? 0,
      contour: pitch?.contour ?? [],
      envelope: pitch?.envelope ?? [],
      top: s.top.map((t) => ({ code: LABELS[t.index].code, name: LABELS[t.index].name, p: Math.round(t.p * 1000) / 1000 })),
      pca: s.pca,
      embedding: Array.from(s.embedding, (v) => Math.round(v * 100) / 100),
    }
  })

const paramCount = trained.weights.reduce((sum, w, i) => {
  if (!w) return sum
  const spec = trained.spec[i]
  const out = spec.type === 'conv' ? spec.cout : spec.type === 'dense' ? spec.outDim : 0
  return sum + w.w.length + out
}, 0)

writeFileSync(
  `${ROOT}/public/clips.json`,
  JSON.stringify({
    generatedAt: new Date().toISOString(),
    model: {
      architecture: 'cnn 64x32 -> conv10/pool -> conv20/pool -> 64 -> 21',
      classes: trained.classes.length,
      valAccuracy: Math.round(trained.valAccuracy * 1000) / 1000,
      params: paramCount,
      perClass: trained.perClass,
    },
    dsp: {
      sampleRate: DEFAULTS.sampleRate,
      nFft: DEFAULTS.nFft,
      hop: DEFAULTS.hop,
      melBands: DEFAULTS.nMels,
      poolFrames: DEFAULTS.poolFrames,
      windowSeconds: DEFAULTS.windowSeconds,
      dim: FEATURE_DIM,
    },
    clips: clipData,
  }),
)
console.log(`wrote ${clipData.length} clips -> public/clips/ + public/clips.json`)

// ------------------------------------------------- quantise for the Worker ---
const quantisedLayers: QuantisedLayer[] = trained.spec.map((spec, i) => {
  const w = trained.weights[i]
  if (!w || spec.type === 'pool') return null
  const outDim = spec.type === 'conv' ? spec.cout : spec.outDim
  const inDim = w.w.length / outDim
  const scales = new Array(outDim)
  const int8 = new Int8Array(w.w.length)
  for (let o = 0; o < outDim; o++) {
    let max = 0
    for (let k = 0; k < inDim; k++) max = Math.max(max, Math.abs(w.w[o * inDim + k]))
    const scale = max / 127 || 1e-8
    scales[o] = scale
    for (let k = 0; k < inDim; k++) {
      int8[o * inDim + k] = Math.max(-127, Math.min(127, Math.round(w.w[o * inDim + k] / scale)))
    }
  }
  return {
    w: Buffer.from(int8.buffer, int8.byteOffset, int8.byteLength).toString('base64'),
    s: scales,
    b: w.b,
  }
})

const quantised: QuantisedModel = {
  inputH: trained.inputH,
  inputW: trained.inputW,
  classes: trained.classes,
  spec: trained.spec,
  layers: quantisedLayers,
}

const ts = `// generated by tools/export.ts - do not edit
import type { QuantisedModel } from './model.ts'

export const STATS = {
  mean: ${JSON.stringify(stats.mean.map((v) => Math.round(v * 1e4) / 1e4))},
  std: ${JSON.stringify(stats.std.map((v) => Math.round(v * 1e4) / 1e4))},
}

export const MODEL_DATA: QuantisedModel = ${JSON.stringify(quantised)}
`
writeFileSync(`${ROOT}/src/model-data.ts`, ts)
console.log(`quantised model -> src/model-data.ts (${(ts.length / 1024).toFixed(0)} kB, ${paramCount} params)`)

