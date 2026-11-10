/**
 * model.ts - the deployed network: a small CNN over the 64 x 32 log-mel image.
 *
 *   input 1 x 64 x 32
 *   conv 3x3, 16 filters, pad 1, ReLU -> maxpool 2  (16 x 32 x 16)
 *   conv 3x3, 32 filters, pad 1, ReLU -> maxpool 2  (32 x 16 x 8)
 *   flatten 4096 -> dense 64 ReLU -> dense 21
 *
 * Weights ship as int8 with a per-output-row scale, so the whole net is about
 * 270 kB of base64 inside the Worker bundle and inference is sub-millisecond.
 */

export type Spec =
  | { type: 'conv'; cin: number; cout: number; k: number; pad: number; relu: boolean }
  | { type: 'pool'; size: number }
  | { type: 'dense'; inDim: number; outDim: number; relu: boolean }

export type Tensor = { data: Float32Array; c: number; h: number; w: number }

export type Weights = { w: Float32Array; b: Float32Array }

export type Model = {
  inputH: number
  inputW: number
  spec: Spec[]
  weights: (Weights | null)[]
  classes: string[]
}

export type QuantisedLayer = { w: string; s: number[]; b: number[] } | null

export type QuantisedModel = {
  inputH: number
  inputW: number
  classes: string[]
  spec: Spec[]
  layers: QuantisedLayer[]
}

/** Packs a 2048-float feature window (band-major 64 x 32) into a 1 x 64 x 32 tensor. */
export function toTensor(x: Float32Array, h = 64, w = 32): Tensor {
  return { data: x, c: 1, h, w }
}

export function conv2d(x: Tensor, spec: Extract<Spec, { type: 'conv' }>, w: Float32Array, b: Float32Array): Tensor {
  const { cout, cin, k, pad, relu } = spec
  const outH = x.h + 2 * pad - k + 1
  const outW = x.w + 2 * pad - k + 1
  const out = new Float32Array(cout * outH * outW)
  const kk = k * k

  for (let o = 0; o < cout; o++) {
    const wBase = o * cin * kk
    for (let y = 0; y < outH; y++) {
      for (let xx = 0; xx < outW; xx++) {
        let acc = b[o]
        for (let c = 0; c < cin; c++) {
          const xBase = c * x.h * x.w
          const wc = wBase + c * kk
          for (let ky = 0; ky < k; ky++) {
            const iy = y + ky - pad
            if (iy < 0 || iy >= x.h) continue
            for (let kx = 0; kx < k; kx++) {
              const ix = xx + kx - pad
              if (ix < 0 || ix >= x.w) continue
              acc += w[wc + ky * k + kx] * x.data[xBase + iy * x.w + ix]
            }
          }
        }
        out[o * outH * outW + y * outW + xx] = relu && acc < 0 ? 0 : acc
      }
    }
  }
  return { data: out, c: cout, h: outH, w: outW }
}

export function maxPool(x: Tensor, size: number): Tensor {
  const outH = Math.floor(x.h / size)
  const outW = Math.floor(x.w / size)
  const out = new Float32Array(x.c * outH * outW)
  for (let c = 0; c < x.c; c++) {
    for (let y = 0; y < outH; y++) {
      for (let xx = 0; xx < outW; xx++) {
        let best = -Infinity
        for (let dy = 0; dy < size; dy++) {
          for (let dx = 0; dx < size; dx++) {
            const v = x.data[c * x.h * x.w + (y * size + dy) * x.w + xx * size + dx]
            if (v > best) best = v
          }
        }
        out[c * outH * outW + y * outW + xx] = best
      }
    }
  }
  return { data: out, c: x.c, h: outH, w: outW }
}

export function dense(x: Float32Array, spec: Extract<Spec, { type: 'dense' }>, w: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(spec.outDim)
  for (let o = 0; o < spec.outDim; o++) {
    let acc = b[o]
    const row = o * spec.inDim
    for (let i = 0; i < spec.inDim; i++) acc += w[row + i] * x[i]
    out[o] = spec.relu && acc < 0 ? 0 : acc
  }
  return out
}

type Activation = { tensor: Tensor; flat?: Float32Array }

/**
 * Runs the network. `activations` is filled when `keep` is true, which the
 * Trainer uses for backprop; the Worker never needs it.
 */
export function forward(model: Model, input: Float32Array, keep = false): { logits: Float32Array; embedding: Float32Array; acts: (Tensor | Float32Array)[] } {
  let t: Tensor = toTensor(input, model.inputH, model.inputW)
  const acts: (Tensor | Float32Array)[] = [t]

  for (let i = 0; i < model.spec.length; i++) {
    const spec = model.spec[i]
    const weights = model.weights[i]
    if (spec.type === 'conv') {
      t = conv2d(t, spec, weights!.w, weights!.b)
      acts.push(t)
    } else if (spec.type === 'pool') {
      t = maxPool(t, spec.size)
      acts.push(t)
    } else {
      const out = dense(t.data, spec, weights!.w, weights!.b)
      acts.push(out)
      t = { data: out, c: 1, h: 1, w: out.length }
    }
  }

  const logits = t.data as Float32Array
  // embedding = last hidden dense (the one before the classifier)
  let embedding = new Float32Array(0)
  for (let i = model.spec.length - 1; i >= 0; i--) {
    const spec = model.spec[i]
    if (spec.type === 'dense' && spec.relu) {
      embedding = acts[i + 1] as Float32Array
      break
    }
  }
  return { logits, embedding, acts }
}

export function softmax(logits: Float32Array | number[]): number[] {
  let max = -Infinity
  for (const v of logits) max = Math.max(max, v)
  const exps = Array.from(logits, (v) => Math.exp(v - max))
  const sum = exps.reduce((a, b) => a + b, 0)
  return exps.map((v) => v / sum)
}

export type Prediction = {
  probs: number[]
  logits: Float32Array
  embedding: Float32Array
  top: { index: number; p: number }[]
}

export function predict(model: Model, x: Float32Array): Prediction {
  const { logits, embedding } = forward(model, x, false)
  const probs = softmax(logits)
  const top = probs
    .map((p, index) => ({ index, p }))
    .sort((a, b) => b.p - a.p)
    .slice(0, 3)
  return { probs, logits, embedding, top }
}

export function predictWindows(model: Model, windows: Float32Array[]): Prediction {
  let sum: number[] | null = null
  let last: Prediction | null = null
  for (const w of windows) {
    const p = predict(model, w)
    last = p
    if (!sum) sum = p.probs.slice()
    else for (let i = 0; i < sum.length; i++) sum[i] += p.probs[i]
  }
  if (!sum || !last) throw new Error('no windows')
  const probs = sum.map((v) => v / windows.length)
  const top = probs
    .map((p, index) => ({ index, p }))
    .sort((a, b) => b.p - a.p)
    .slice(0, 3)
  return { ...last, probs, top }
}

// ------------------------------------------------------------- quantisation --
function decodeBase64Int8(input: string): Int8Array {
  const bin = atob(input)
  const out = new Int8Array(bin.length)
  for (let i = 0; i < bin.length; i++) {
    const v = bin.charCodeAt(i)
    out[i] = v > 127 ? v - 256 : v
  }
  return out
}

export function loadModel(q: QuantisedModel): Model {
  const weights = q.layers.map((layer) => {
    if (!layer) return null
    const int8 = decodeBase64Int8(layer.w)
    const w = new Float32Array(int8.length)
    const outDim = layer.s.length
    const inDim = int8.length / outDim
    for (let o = 0; o < outDim; o++) {
      const scale = layer.s[o]
      const row = o * inDim
      for (let i = 0; i < inDim; i++) w[row + i] = int8[row + i] * scale
    }
    return { w, b: Float32Array.from(layer.b) }
  })
  return { inputH: q.inputH, inputW: q.inputW, spec: q.spec, weights, classes: q.classes }
}
