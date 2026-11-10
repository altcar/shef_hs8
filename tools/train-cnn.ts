/**
 * train-cnn.ts - trains the deployed CNN on the cached DSP features.
 *
 *   conv 3x3 -> pool -> conv 3x3 -> pool -> dense 64 -> dense 21
 *
 * Forward and backward are hand-written direct convolutions (no im2col, no
 * native deps) so the whole thing runs on plain Node.
 *
 * Usage: node tools/train-cnn.ts [--epochs 25] [--c1 12] [--c2 24]
 */
import { readFileSync, writeFileSync } from 'node:fs'
import { DEFAULTS, FEATURE_DIM } from '../src/dsp.ts'
import { softmax, type Spec } from '../src/model.ts'
import { LABELS, LABEL_INDEX } from '../src/species.ts'

const CACHE = 'C:/Users/c85-lau/AppData/Local/Temp/opencode/birds/wizzibiz/cache'
const H = DEFAULTS.nMels // 64 bands
const W = DEFAULTS.poolFrames // 32 frames

function arg(name: string, fallback: number): number {
  const i = process.argv.indexOf(`--${name}`)
  return i > -1 ? Number(process.argv[i + 1]) : fallback
}

const EPOCHS = arg('epochs', 25)
const BATCH = arg('batch', 64)
const LR = arg('lr', 0.002)
const C1 = arg('c1', 12)
const C2 = arg('c2', 24)
const SEED = arg('seed', 20261110)

// ---------------------------------------------------------------- data ------
const meta = JSON.parse(readFileSync(`${CACHE}/meta.json`, 'utf8')) as {
  windows: number
  clips: { id: string; label: string }[]
}
const stats = JSON.parse(readFileSync(`${CACHE}/stats.json`, 'utf8')) as { mean: number[]; std: number[] }
const raw = readFileSync(`${CACHE}/raw.f32`)
const feats = new Float32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength))

const WINDOWS = meta.windows
const N = meta.clips.length
const NC = LABELS.length
const mean = Float32Array.from(stats.mean)
const std = Float32Array.from(stats.std)

for (let i = 0; i < feats.length; i++) {
  const band = Math.floor((i % FEATURE_DIM) / DEFAULTS.poolFrames)
  feats[i] = (feats[i] - mean[band]) / (std[band] || 1)
}

const y = new Int32Array(N)
for (let i = 0; i < N; i++) y[i] = LABEL_INDEX.get(meta.clips[i].label) ?? NC - 1

let seed = SEED
function rand(): number {
  seed |= 0
  seed = (seed + 0x6d2b79f5) | 0
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296
}

const trainIdx: number[] = []
const valIdx: number[] = []
for (let c = 0; c < NC; c++) {
  const idx: number[] = []
  for (let i = 0; i < N; i++) if (y[i] === c) idx.push(i)
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1))
    ;[idx[i], idx[j]] = [idx[j], idx[i]]
  }
  const cut = Math.max(1, Math.round(idx.length * 0.15))
  valIdx.push(...idx.slice(0, cut))
  trainIdx.push(...idx.slice(cut))
}

const classCount = new Array(NC).fill(0)
for (let i = 0; i < N; i++) classCount[y[i]]++
const classWeight = Float32Array.from(classCount, (n) => Math.sqrt(N / (NC * (n || 1))))
{
  let sum = 0
  for (const w of classWeight) sum += w
  for (let c = 0; c < NC; c++) classWeight[c] /= sum / NC
}

const POOL2_H = Math.floor(H / 4)
const POOL2_W = Math.floor(W / 4)
const FLAT = C2 * POOL2_H * POOL2_W

console.log(`clips ${N} (${trainIdx.length} train / ${valIdx.length} val), classes ${NC}, conv ${C1}/${C2}, flat ${FLAT}`)

// --------------------------------------------------------------- layers -----
const SPEC: Spec[] = [
  { type: 'conv', cin: 1, cout: C1, k: 3, pad: 1, relu: true },
  { type: 'pool', size: 2 },
  { type: 'conv', cin: C1, cout: C2, k: 3, pad: 1, relu: true },
  { type: 'pool', size: 2 },
  { type: 'dense', inDim: FLAT, outDim: 64, relu: true },
  { type: 'dense', inDim: 64, outDim: NC, relu: false },
]

type Params = { w: Float32Array; b: Float32Array }
const params: (Params | null)[] = SPEC.map((s) => {
  if (s.type === 'conv') {
    const size = s.cout * s.cin * s.k * s.k
    const w = new Float32Array(size)
    const scale = Math.sqrt(2 / (s.cin * s.k * s.k))
    for (let i = 0; i < size; i++) w[i] = (rand() * 2 - 1) * scale
    return { w, b: new Float32Array(s.cout) }
  }
  if (s.type === 'dense') {
    const w = new Float32Array(s.outDim * s.inDim)
    const scale = Math.sqrt(2 / s.inDim)
    for (let i = 0; i < w.length; i++) w[i] = (rand() * 2 - 1) * scale
    return { w, b: new Float32Array(s.outDim) }
  }
  return null
})

const gradP = params.map((p): Params => ({ w: new Float32Array(p?.w.length ?? 0), b: new Float32Array(p?.b.length ?? 0) }))
const mW = params.map((p) => new Float32Array(p?.w.length ?? 0))
const vW = params.map((p) => new Float32Array(p?.w.length ?? 0))
const mB = params.map((p) => new Float32Array(p?.b.length ?? 0))
const vB = params.map((p) => new Float32Array(p?.b.length ?? 0))

// ------------------------------------------------------- forward / back -----
type Node = { kind: 'input' | 'conv' | 'pool' | 'dense'; data: Float32Array; c: number; h: number; w: number; argmax?: Int32Array }

function convForward(x: Float32Array, inC: number, inH: number, inW: number, s: Extract<Spec, { type: 'conv' }>, p: Params) {
  const outH = inH + 2 * s.pad - s.k + 1
  const outW = inW + 2 * s.pad - s.k + 1
  const out = new Float32Array(s.cout * outH * outW)
  const kk = s.k * s.k
  for (let o = 0; o < s.cout; o++) {
    const wBase = o * s.cin * kk
    const oBase = o * outH * outW
    for (let yy = 0; yy < outH; yy++) {
      for (let xx = 0; xx < outW; xx++) {
        let acc = p.b[o]
        for (let c = 0; c < s.cin; c++) {
          const wc = wBase + c * kk
          const xc = c * inH * inW
          for (let ky = 0; ky < s.k; ky++) {
            const iy = yy + ky - s.pad
            if (iy < 0 || iy >= inH) continue
            for (let kx = 0; kx < s.k; kx++) {
              const ix = xx + kx - s.pad
              if (ix < 0 || ix >= inW) continue
              acc += p.w[wc + ky * s.k + kx] * x[xc + iy * inW + ix]
            }
          }
        }
        out[oBase + yy * outW + xx] = s.relu && acc < 0 ? 0 : acc
      }
    }
  }
  return { out, outH, outW }
}

function poolForward(x: Float32Array, c: number, h: number, w: number, size: number) {
  const outH = Math.floor(h / size)
  const outW = Math.floor(w / size)
  const out = new Float32Array(c * outH * outW)
  const argmax = new Int32Array(c * outH * outW)
  for (let ch = 0; ch < c; ch++) {
    for (let yy = 0; yy < outH; yy++) {
      for (let xx = 0; xx < outW; xx++) {
        let best = -Infinity
        let arg = 0
        for (let dy = 0; dy < size; dy++) {
          for (let dx = 0; dx < size; dx++) {
            const idx = ch * h * w + (yy * size + dy) * w + xx * size + dx
            if (x[idx] > best) {
              best = x[idx]
              arg = idx
            }
          }
        }
        out[ch * outH * outW + yy * outW + xx] = best
        argmax[ch * outH * outW + yy * outW + xx] = arg
      }
    }
  }
  return { out, argmax, outH, outW }
}

function forward(x: Float32Array) {
  const nodes: Node[] = [{ kind: 'input', data: x, c: 1, h: H, w: W }]
  for (let i = 0; i < SPEC.length; i++) {
    const s = SPEC[i]
    const prev = nodes[nodes.length - 1]
    const p = params[i]!
    if (s.type === 'conv') {
      const r = convForward(prev.data, prev.c, prev.h, prev.w, s, p)
      nodes.push({ kind: 'conv', data: r.out, c: s.cout, h: r.outH, w: r.outW })
    } else if (s.type === 'pool') {
      const r = poolForward(prev.data, prev.c, prev.h, prev.w, s.size)
      nodes.push({ kind: 'pool', data: r.out, c: prev.c, h: r.outH, w: r.outW, argmax: r.argmax })
    } else {
      const out = new Float32Array(s.outDim)
      for (let o = 0; o < s.outDim; o++) {
        let acc = p.b[o]
        const row = o * s.inDim
        for (let k = 0; k < s.inDim; k++) acc += p.w[row + k] * prev.data[k]
        out[o] = s.relu && acc < 0 ? 0 : acc
      }
      nodes.push({ kind: 'dense', data: out, c: 1, h: 1, w: out.length })
    }
  }
  const logits = nodes[nodes.length - 1].data
  return { nodes, logits, probs: softmax(logits) }
}

function backward(f: ReturnType<typeof forward>, label: number, weight: number) {
  let d = new Float32Array(f.logits.length)
  for (let i = 0; i < d.length; i++) d[i] = (f.probs[i] - (i === label ? 1 : 0)) * weight

  for (let i = SPEC.length - 1; i >= 0; i--) {
    const s = SPEC[i]
    const out = f.nodes[i + 1]
    const input = f.nodes[i]
    const p = params[i]!

    if (s.type === 'dense') {
      const g = gradP[i]
      for (let o = 0; o < s.outDim; o++) {
        const dv = d[o]
        if (dv === 0) continue
        const row = o * s.inDim
        g.b[o] += dv
        for (let k = 0; k < s.inDim; k++) g.w[row + k] += dv * input.data[k]
      }
      const dIn = new Float32Array(s.inDim)
      for (let o = 0; o < s.outDim; o++) {
        const dv = d[o]
        if (dv === 0) continue
        const row = o * s.inDim
        for (let k = 0; k < s.inDim; k++) dIn[k] += dv * p.w[row + k]
      }
      // ReLU derivative: the logits layer's input is the hidden dense layer
      if (i === SPEC.length - 1) {
        for (let k = 0; k < dIn.length; k++) if (input.data[k] <= 0) dIn[k] = 0
      }
      d = dIn
      continue
    }

    if (s.type === 'pool') {
      const dIn = new Float32Array(input.data.length)
      const argmax = out.argmax!
      for (let k = 0; k < d.length; k++) dIn[argmax[k]] += d[k]
      d = dIn
      continue
    }

    // conv: d is the gradient of the (post-ReLU) output
    const reluMask = out.data
    for (let k = 0; k < d.length; k++) if (reluMask[k] <= 0) d[k] = 0

    const g = gradP[i]
    const { c: inC, h: inH, w: inW } = input
    const outH = out.h
    const outW = out.w
    const kk = s.k * s.k
    const dIn = new Float32Array(input.data.length)

    for (let o = 0; o < s.cout; o++) {
      const wBase = o * s.cin * kk
      const oBase = o * outH * outW
      for (let yy = 0; yy < outH; yy++) {
        for (let xx = 0; xx < outW; xx++) {
          const dv = d[oBase + yy * outW + xx]
          if (dv === 0) continue
          g.b[o] += dv
          for (let c = 0; c < s.cin; c++) {
            const wc = wBase + c * kk
            const xc = c * inH * inW
            for (let ky = 0; ky < s.k; ky++) {
              const iy = yy + ky - s.pad
              if (iy < 0 || iy >= inH) continue
              for (let kx = 0; kx < s.k; kx++) {
                const ix = xx + kx - s.pad
                if (ix < 0 || ix >= inW) continue
                g.w[wc + ky * s.k + kx] += dv * input.data[xc + iy * inW + ix]
                dIn[xc + iy * inW + ix] += dv * p.w[wc + ky * s.k + kx]
              }
            }
          }
        }
      }
    }
    d = dIn
    void inC
  }
}

// ------------------------------------------------------------- training -----
const B1 = 0.9
const B2 = 0.999
const EPS = 1e-8
let step = 0

function adam() {
  step++
  const bc1 = 1 - Math.pow(B1, step)
  const bc2 = 1 - Math.pow(B2, step)
  for (let i = 0; i < params.length; i++) {
    const p = params[i]
    if (!p) continue
    const g = gradP[i]
    for (let k = 0; k < p.w.length; k++) {
      const grad = g.w[k]
      mW[i][k] = B1 * mW[i][k] + (1 - B1) * grad
      vW[i][k] = B2 * vW[i][k] + (1 - B2) * grad * grad
      p.w[k] -= (LR * (mW[i][k] / bc1)) / (Math.sqrt(vW[i][k] / bc2) + EPS)
    }
    for (let k = 0; k < p.b.length; k++) {
      const grad = g.b[k]
      mB[i][k] = B1 * mB[i][k] + (1 - B1) * grad
      vB[i][k] = B2 * vB[i][k] + (1 - B2) * grad * grad
      p.b[k] -= (LR * (mB[i][k] / bc1)) / (Math.sqrt(vB[i][k] / bc2) + EPS)
    }
  }
}

/** one training sample: random window + circular time roll + band mask + noise */
function sample(clip: number): Float32Array {
  const wIdx = Math.floor(rand() * WINDOWS)
  const base = clip * WINDOWS * FEATURE_DIM + wIdx * FEATURE_DIM
  const out = new Float32Array(FEATURE_DIM)
  const roll = Math.floor((rand() * 2 - 1) * 6)
  const noise = (rand() * 2 - 1) * 0.03
  const maskStart = rand() < 0.3 ? Math.floor(rand() * 56) : -1
  for (let b = 0; b < H; b++) {
    for (let t = 0; t < W; t++) {
      const src = base + b * W + ((((t + roll) % W) + W) % W)
      let val = feats[src] + noise
      if (maskStart >= 0 && b >= maskStart && b < maskStart + 8) val = 0
      out[b * W + t] = val
    }
  }
  return out
}

function evaluate(indices: number[]): { acc: number; confusion: number[][] } {
  let correct = 0
  const confusion = Array.from({ length: NC }, () => new Array(NC).fill(0))
  for (const clip of indices) {
    const x = new Float32Array(FEATURE_DIM)
    const base = clip * WINDOWS * FEATURE_DIM
    x.set(feats.subarray(base, base + FEATURE_DIM))
    const f = forward(x)
    let best = 0
    for (let c = 1; c < NC; c++) if (f.probs[c] > f.probs[best]) best = c
    if (best === y[clip]) correct++
    confusion[y[clip]][best]++
  }
  return { acc: correct / indices.length, confusion }
}

const t0 = Date.now()
let bestAcc = -1
let bestState: string | null = null

for (let epoch = 1; epoch <= EPOCHS; epoch++) {
  for (let i = trainIdx.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1))
    ;[trainIdx[i], trainIdx[j]] = [trainIdx[j], trainIdx[i]]
  }

  let loss = 0
  for (let start = 0; start < trainIdx.length; start += BATCH) {
    for (let i = 0; i < gradP.length; i++) {
      gradP[i].w.fill(0)
      gradP[i].b.fill(0)
    }
    const batch = trainIdx.slice(start, start + BATCH)
    for (const clip of batch) {
      const x = sample(clip)
      const f = forward(x)
      loss += -Math.log(Math.max(1e-9, f.probs[y[clip]]))
      backward(f, y[clip], classWeight[y[clip]] / batch.length)
    }
    adam()
  }

  const trainAcc = evaluate(trainIdx.slice(0, Math.min(300, trainIdx.length))).acc
  const { acc: valAcc } = evaluate(valIdx)
  console.log(
    `epoch ${String(epoch).padStart(2)}  loss ${(loss / trainIdx.length).toFixed(3)}  train ${(trainAcc * 100).toFixed(1)}%  val ${(valAcc * 100).toFixed(1)}%  ${((Date.now() - t0) / 1000).toFixed(0)}s`,
  )

  if (valAcc > bestAcc) {
    bestAcc = valAcc
    bestState = JSON.stringify({ weights: params.map((p) => (p ? { w: Array.from(p.w), b: Array.from(p.b) } : null)) })
  }
}

if (!bestState) throw new Error('no weights')
const parsed = JSON.parse(bestState) as { weights: { w: number[]; b: number[] }[] }
for (let i = 0; i < params.length; i++) {
  const p = params[i]
  if (!p) continue
  p.w.set(parsed.weights[i].w)
  p.b.set(parsed.weights[i].b)
}

const { acc, confusion } = evaluate(valIdx)
const perClass = confusion.map((row, c) => {
  const total = row.reduce((a, b) => a + b, 0)
  return { code: LABELS[c].code, name: LABELS[c].name, total, recall: total ? row[c] / total : 0 }
})
perClass.sort((a, b) => b.recall - a.recall)
console.log('\nper-class recall (val):')
for (const c of perClass) console.log(`  ${(c.recall * 100).toFixed(0).padStart(3)}%  ${c.code.padEnd(9)} ${c.name}  (n=${c.total})`)

writeFileSync(
  `${CACHE}/model.json`,
  JSON.stringify({
    inputH: H,
    inputW: W,
    spec: SPEC,
    classes: LABELS.map((l) => l.code),
    weights: params.map((p) => (p ? { w: Array.from(p.w), b: Array.from(p.b) } : null)),
    valAccuracy: acc,
    perClass,
    confusion,
  }),
)
console.log(`\nbest val accuracy: ${(acc * 100).toFixed(1)}% in ${((Date.now() - t0) / 1000).toFixed(0)}s -> model.json`)
