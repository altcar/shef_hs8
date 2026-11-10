/**
 * train.ts - trains the species classifier on the DSP features.
 *
 * 2048 -> 96 -> 64 -> 21 (20 species + "not a bird"), ReLU, softmax,
 * Adam, mini-batches, one randomly chosen analysis window per clip per epoch
 * (that is our time-shift augmentation) plus gaussian feature noise.
 *
 * Usage: node tools/train.ts [--epochs 30] [--batch 64] [--lr 0.001]
 */
import { readFileSync, writeFileSync } from 'node:fs'
import { DEFAULTS, FEATURE_DIM } from '../src/dsp.ts'
import { HIDDEN, INPUT_DIM, softmax } from '../src/model.ts'
import { LABELS, LABEL_INDEX } from '../src/species.ts'

const CACHE = 'C:/Users/c85-lau/AppData/Local/Temp/opencode/birds/wizzibiz/cache'

function arg(name: string, fallback: number): number {
  const i = process.argv.indexOf(`--${name}`)
  return i > -1 ? Number(process.argv[i + 1]) : fallback
}

const EPOCHS = arg('epochs', 30)
const BATCH = arg('batch', 64)
const LR = arg('lr', 0.002)
const NOISE = arg('noise', 0.08)
const SEED = arg('seed', 1337)

// ---------------------------------------------------------------- data ------
const meta = JSON.parse(readFileSync(`${CACHE}/meta.json`, 'utf8')) as {
  windows: number
  dim: number
  clips: { id: string; label: string }[]
}
const stats = JSON.parse(readFileSync(`${CACHE}/stats.json`, 'utf8')) as { mean: number[]; std: number[] }
const raw = readFileSync(`${CACHE}/raw.f32`)
const rawCopy = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength)
const feats = new Float32Array(rawCopy)

const WINDOWS = meta.windows
const N = meta.clips.length
const CLASSES = LABELS.length

// standardise in place: layouts are band-major inside each 2048-float window
const mean = Float32Array.from(stats.mean)
const std = Float32Array.from(stats.std)
for (let i = 0; i < feats.length; i++) {
  const band = Math.floor((i % FEATURE_DIM) / DEFAULTS.poolFrames)
  feats[i] = (feats[i] - mean[band]) / (std[band] || 1)
}

const y = new Int32Array(N)
for (let i = 0; i < N; i++) y[i] = LABEL_INDEX.get(meta.clips[i].label) ?? CLASSES - 1

// stratified 85/15 split by clip
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
for (let c = 0; c < CLASSES; c++) {
  const idx = []
  for (let i = 0; i < N; i++) if (y[i] === c) idx.push(i)
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1))
    ;[idx[i], idx[j]] = [idx[j], idx[i]]
  }
  const cut = Math.max(1, Math.round(idx.length * 0.15))
  valIdx.push(...idx.slice(0, cut))
  trainIdx.push(...idx.slice(cut))
}

console.log(`clips: ${N} (${trainIdx.length} train / ${valIdx.length} val), classes: ${CLASSES}`)

// inverse-frequency class weights, softened with a square root so a big
// negative class does not dominate the loss
const classCount = new Array(CLASSES).fill(0)
for (let i = 0; i < N; i++) classCount[y[i]]++
const classWeight = Float32Array.from(classCount, (n) => Math.sqrt(N / (CLASSES * (n || 1))))
{
  let sum = 0
  for (const w of classWeight) sum += w
  const meanW = sum / CLASSES
  for (let c = 0; c < CLASSES; c++) classWeight[c] /= meanW
  console.log('class weights:', Array.from(classWeight, (w) => w.toFixed(2)).join(' '))
}

// --------------------------------------------------------------- model ------
type Layer = { inDim: number; outDim: number; w: Float32Array; b: Float32Array }
const dims = [INPUT_DIM, ...HIDDEN, CLASSES]
const layers: Layer[] = []
const mW: Float32Array[] = []
const vW: Float32Array[] = []
const mB: Float32Array[] = []
const vB: Float32Array[] = []

for (let l = 0; l < dims.length - 1; l++) {
  const inDim = dims[l]
  const outDim = dims[l + 1]
  const w = new Float32Array(outDim * inDim)
  const b = new Float32Array(outDim)
  const scale = Math.sqrt(2 / inDim) // He init
  for (let i = 0; i < w.length; i++) w[i] = (rand() * 2 - 1) * scale
  layers.push({ inDim, outDim, w, b })
  mW.push(new Float32Array(w.length)); vW.push(new Float32Array(w.length))
  mB.push(new Float32Array(outDim)); vB.push(new Float32Array(outDim))
}

const B1 = 0.9, B2 = 0.999, EPS = 1e-8
let step = 0

function forward(x: Float32Array, keep: boolean): { acts: Float32Array[]; probs: number[] } {
  const acts: Float32Array[] = [x]
  let h = x
  for (let l = 0; l < layers.length; l++) {
    const layer = layers[l]
    const out = new Float32Array(layer.outDim)
    const last = l === layers.length - 1
    for (let o = 0; o < layer.outDim; o++) {
      let acc = layer.b[o]
      const row = o * layer.inDim
      for (let i = 0; i < layer.inDim; i++) acc += layer.w[row + i] * h[i]
      out[o] = last || acc > 0 ? acc : 0
    }
    acts.push(out)
    h = out
  }
  return { acts, probs: softmax(h) }
}

function accuracy(indices: number[]): number {
  let correct = 0
  for (const i of indices) {
    const { probs } = forward(feats.subarray(i * WINDOWS * FEATURE_DIM, i * WINDOWS * FEATURE_DIM + FEATURE_DIM), false)
    let best = 0
    for (let c = 1; c < probs.length; c++) if (probs[c] > probs[best]) best = c
    if (best === y[i]) correct++
  }
  return correct / indices.length
}

const t0 = Date.now()
let bestAcc = 0
let bestWeights: { w: Float32Array[]; b: Float32Array[] } | null = null

for (let epoch = 1; epoch <= EPOCHS; epoch++) {
  for (let i = trainIdx.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1))
    ;[trainIdx[i], trainIdx[j]] = [trainIdx[j], trainIdx[i]]
  }

  let loss = 0
  for (let start = 0; start < trainIdx.length; start += BATCH) {
    const batch = trainIdx.slice(start, start + BATCH)
    const gradW = layers.map((l) => new Float32Array(l.w.length))
    const gradB = layers.map((l) => new Float32Array(l.b.length))

    for (const clip of batch) {
      // augmentation: random window + gaussian noise
      const wIdx = Math.floor(rand() * WINDOWS)
      const base = clip * WINDOWS * FEATURE_DIM + wIdx * FEATURE_DIM
      const x = new Float32Array(FEATURE_DIM)
      for (let i = 0; i < FEATURE_DIM; i++) {
        const n = Math.random() < 0.5 ? 0 : 1
        x[i] = feats[base + i] + (rand() * 2 - 1) * NOISE * n
      }

      const { acts, probs } = forward(x, true)
      loss += -Math.log(Math.max(1e-9, probs[y[clip]]))

      // backprop: dL/dlogits = (probs - onehot) * class weight
      let delta = new Float32Array(probs.length)
      const cw = classWeight[y[clip]]
      for (let c = 0; c < probs.length; c++) delta[c] = (probs[c] - (c === y[clip] ? 1 : 0)) * cw

      for (let l = layers.length - 1; l >= 0; l--) {
        const layer = layers[l]
        const input = acts[l]
        const gW = gradW[l]
        const gB = gradB[l]
        for (let o = 0; o < layer.outDim; o++) {
          const d = delta[o]
          if (d === 0) continue
          gB[o] += d
          const row = o * layer.inDim
          for (let i = 0; i < layer.inDim; i++) gW[row + i] += d * input[i]
        }
        if (l > 0) {
          const prev = new Float32Array(layer.inDim)
          for (let i = 0; i < layer.inDim; i++) {
            let acc = 0
            for (let o = 0; o < layer.outDim; o++) acc += delta[o] * layer.w[o * layer.inDim + i]
            prev[i] = acc
          }
          // ReLU derivative of the previous layer's activations
          for (let i = 0; i < prev.length; i++) if (acts[l][i] <= 0) prev[i] = 0
          delta = prev
        }
      }
    }

    // Adam step
    step++
    const scale = 1 / batch.length
    const bc1 = 1 - Math.pow(B1, step)
    const bc2 = 1 - Math.pow(B2, step)
    for (let l = 0; l < layers.length; l++) {
      const layer = layers[l]
      for (let i = 0; i < layer.w.length; i++) {
        const g = gradW[l][i] * scale
        mW[l][i] = B1 * mW[l][i] + (1 - B1) * g
        vW[l][i] = B2 * vW[l][i] + (1 - B2) * g * g
        layer.w[i] -= (LR * (mW[l][i] / bc1)) / (Math.sqrt(vW[l][i] / bc2) + EPS)
      }
      for (let i = 0; i < layer.b.length; i++) {
        const g = gradB[l][i] * scale
        mB[l][i] = B1 * mB[l][i] + (1 - B1) * g
        vB[l][i] = B2 * vB[l][i] + (1 - B2) * g * g
        layer.b[i] -= (LR * (mB[l][i] / bc1)) / (Math.sqrt(vB[l][i] / bc2) + EPS)
      }
    }
  }

  const trainAcc = accuracy(trainIdx.slice(0, Math.min(600, trainIdx.length)))
  const valAcc = accuracy(valIdx)
  const elapsed = ((Date.now() - t0) / 1000).toFixed(0)
  console.log(
    `epoch ${String(epoch).padStart(2)}  loss ${(loss / trainIdx.length).toFixed(3)}  train ${(trainAcc * 100).toFixed(1)}%  val ${(valAcc * 100).toFixed(1)}%  ${elapsed}s`,
  )

  if (valAcc > bestAcc) {
    bestAcc = valAcc
    bestWeights = { w: layers.map((l) => l.w.slice()), b: layers.map((l) => l.b.slice()) }
  }
}

if (!bestWeights) throw new Error('training produced no weights')
for (let l = 0; l < layers.length; l++) {
  layers[l].w = bestWeights.w[l]
  layers[l].b = bestWeights.b[l]
}

// ------------------------------------------------------------- reporting ----
const confusion = Array.from({ length: CLASSES }, () => new Array(CLASSES).fill(0))
for (const i of valIdx) {
  const { probs } = forward(feats.subarray(i * WINDOWS * FEATURE_DIM, i * WINDOWS * FEATURE_DIM + FEATURE_DIM), false)
  let best = 0
  for (let c = 1; c < probs.length; c++) if (probs[c] > probs[best]) best = c
  confusion[y[i]][best]++
}

const perClass = confusion.map((row, c) => {
  const total = row.reduce((a, b) => a + b, 0)
  return { code: LABELS[c].code, name: LABELS[c].name, total, recall: total ? row[c] / total : 0 }
})
perClass.sort((a, b) => b.recall - a.recall)
console.log('\nper-class recall (val):')
for (const c of perClass) console.log(`  ${(c.recall * 100).toFixed(0).padStart(3)}%  ${c.code.padEnd(9)} ${c.name}  (n=${c.total})`)

const saved = {
  hidden: [...HIDDEN],
  classes: LABELS.map((l) => l.code),
  layers: layers.map((l) => ({ inDim: l.inDim, outDim: l.outDim, w: Array.from(l.w), b: Array.from(l.b) })),
  valAccuracy: bestAcc,
  perClass,
  confusion,
}
writeFileSync(`${CACHE}/model.json`, JSON.stringify(saved))
console.log(`\nbest val accuracy: ${(bestAcc * 100).toFixed(1)}%  -> ${CACHE}/model.json`)

