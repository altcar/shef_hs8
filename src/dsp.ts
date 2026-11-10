/**
 * dsp.ts - the signal-processing core of wizzibiz.
 *
 * Everything here is plain TypeScript with no platform APIs so the same code
 * runs in Node (dataset preprocessing + training) and inside a Cloudflare
 * Worker (live classification), which keeps training and inference features
 * bit-identical.
 *
 * Pipeline for one analysis window:
 *   PCM (any rate) -> resample 16 kHz mono -> Hann-windowed STFT (512/160)
 *   -> 64-band mel filterbank -> log(.) -> time-pool to 32 frames
 *   -> per-dataset standardisation -> 64 * 32 = 2048 dims
 */

export type MonoAudio = { sampleRate: number; samples: Float32Array }

export type StftOptions = {
  nFft?: number
  hop?: number
  nMels?: number
  fMin?: number
  fMax?: number
  poolFrames?: number
}

export const DEFAULTS = {
  sampleRate: 16000,
  nFft: 512,
  hop: 160,
  nMels: 64,
  fMin: 50,
  fMax: 7800,
  windowSeconds: 1,
  poolFrames: 32,
} as const

/** Dimensions of one model input vector. */
export const FEATURE_DIM = DEFAULTS.nMels * DEFAULTS.poolFrames

// ---------------------------------------------------------------------------
// WAV decoding / encoding (16-bit PCM, 8-bit PCM, 24-bit, 32-bit int, float32)
// ---------------------------------------------------------------------------

export function decodeWav(bytes: Uint8Array): MonoAudio {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const riff = String.fromCharCode(...bytes.subarray(0, 4))
  if (riff !== 'RIFF') throw new Error('not a RIFF/WAVE file')

  let offset = 12
  let fmt: { format: number; channels: number; sampleRate: number; bits: number } | null = null
  let dataOffset = -1
  let dataLength = 0

  while (offset + 8 <= bytes.byteLength) {
    const id = String.fromCharCode(...bytes.subarray(offset, offset + 4))
    const size = view.getUint32(offset + 4, true)
    const body = offset + 8
    if (id === 'fmt ') {
      fmt = {
        format: view.getUint16(body, true),
        channels: view.getUint16(body + 2, true),
        sampleRate: view.getUint32(body + 4, true),
        bits: view.getUint16(body + 14, true),
      }
    } else if (id === 'data') {
      dataOffset = body
      dataLength = Math.min(size, bytes.byteLength - body)
      if (fmt) break
    }
    offset = body + size + (size & 1)
  }

  if (!fmt || dataOffset < 0) throw new Error('WAVE file missing fmt/data chunk')

  const bytesPerSample = fmt.bits / 8
  const frameCount = Math.floor(dataLength / (bytesPerSample * fmt.channels))
  const samples = new Float32Array(frameCount)

  for (let i = 0; i < frameCount; i++) {
    let sum = 0
    for (let c = 0; c < fmt.channels; c++) {
      const p = dataOffset + (i * fmt.channels + c) * bytesPerSample
      let v: number
      if (fmt.format === 3 && fmt.bits === 32) v = view.getFloat32(p, true)
      else if (fmt.bits === 8) v = (view.getUint8(p) - 128) / 128
      else if (fmt.bits === 16) v = view.getInt16(p, true) / 32768
      else if (fmt.bits === 24) {
        const b0 = view.getUint8(p)
        const b1 = view.getUint8(p + 1)
        const b2 = view.getInt8(p + 2)
        v = ((b2 << 16) | (b1 << 8) | b0) / 8388608
      } else if (fmt.bits === 32) v = view.getInt32(p, true) / 2147483648
      else throw new Error(`unsupported bit depth ${fmt.bits}`)
      sum += v
    }
    samples[i] = sum / fmt.channels
  }

  return { sampleRate: fmt.sampleRate, samples }
}

export function encodeWav16(samples: Float32Array, sampleRate: number): Uint8Array {
  const bytes = new Uint8Array(44 + samples.length * 2)
  const view = new DataView(bytes.buffer)
  const ascii = (o: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i))
  }
  ascii(0, 'RIFF')
  view.setUint32(4, 36 + samples.length * 2, true)
  ascii(8, 'WAVEfmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, 1, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  ascii(36, 'data')
  view.setUint32(40, samples.length * 2, true)
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true)
  }
  return bytes
}

// ---------------------------------------------------------------------------
// Resampling: windowed-sinc interpolation (works for up- and downsampling)
// ---------------------------------------------------------------------------

const SINC_TAPS = 16

function sinc(x: number): number {
  if (x === 0) return 1
  const px = Math.PI * x
  return Math.sin(px) / px
}

export function resample(samples: Float32Array, from: number, to: number): Float32Array {
  if (from === to) return samples
  const ratio = to / from
  const outLength = Math.max(1, Math.floor(samples.length * ratio))
  const out = new Float32Array(outLength)
  // anti-alias cutoff relative to the *input* Nyquist
  const cutoff = Math.min(1, ratio) * 0.92
  const half = SINC_TAPS

  for (let i = 0; i < outLength; i++) {
    const center = i / ratio
    const base = Math.floor(center)
    let acc = 0
    let norm = 0
    for (let k = -half + 1; k <= half; k++) {
      const idx = base + k
      if (idx < 0 || idx >= samples.length) continue
      const t = center - idx
      const window = 0.5 + 0.5 * Math.cos((Math.PI * t) / half) // Hann
      const w = cutoff * sinc(cutoff * t) * window
      acc += samples[idx] * w
      norm += w
    }
    out[i] = norm !== 0 ? acc / norm : 0
  }
  return out
}

// ---------------------------------------------------------------------------
// STFT (iterative radix-2 complex FFT)
// ---------------------------------------------------------------------------

class Fft {
  readonly n: number
  private readonly cos: Float32Array
  private readonly sin: Float32Array
  private readonly rev: Uint32Array

  constructor(n: number) {
    this.n = n
    this.cos = new Float32Array(n / 2)
    this.sin = new Float32Array(n / 2)
    for (let i = 0; i < n / 2; i++) {
      this.cos[i] = Math.cos((2 * Math.PI * i) / n)
      this.sin[i] = Math.sin((2 * Math.PI * i) / n)
    }
    const bits = Math.log2(n)
    this.rev = new Uint32Array(n)
    for (let i = 0; i < n; i++) {
      let r = 0
      for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b)
      this.rev[i] = r
    }
  }

  /** in-place transform of re/im, length n */
  transform(re: Float32Array, im: Float32Array): void {
    const n = this.n
    for (let i = 0; i < n; i++) {
      const j = this.rev[i]
      if (j > i) {
        let t = re[i]; re[i] = re[j]; re[j] = t
        t = im[i]; im[i] = im[j]; im[j] = t
      }
    }
    for (let size = 2; size <= n; size *= 2) {
      const halfSize = size / 2
      const step = n / size
      for (let i = 0; i < n; i += size) {
        for (let j = i, k = 0; j < i + halfSize; j++, k += step) {
          const l = j + halfSize
          const tpre = re[l] * this.cos[k] + im[l] * this.sin[k]
          const tpim = -re[l] * this.sin[k] + im[l] * this.cos[k]
          re[l] = re[j] - tpre
          im[l] = im[j] - tpim
          re[j] += tpre
          im[j] += tpim
        }
      }
    }
  }
}

const fftCache = new Map<number, Fft>()
function fftFor(n: number): Fft {
  let f = fftCache.get(n)
  if (!f) {
    f = new Fft(n)
    fftCache.set(n, f)
  }
  return f
}

function hann(n: number): Float32Array {
  const w = new Float32Array(n)
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n)
  return w
}

/** Magnitude spectrogram, frames x (nFft/2 + 1). */
export function stftMagnitude(
  samples: Float32Array,
  opts: StftOptions = {},
): { mag: Float32Array; frames: number; bins: number; hop: number } {
  const nFft = opts.nFft ?? DEFAULTS.nFft
  const hop = opts.hop ?? DEFAULTS.hop
  const bins = nFft / 2 + 1
  const frames = Math.max(0, Math.floor((samples.length - nFft) / hop) + 1)
  const mag = new Float32Array(frames * bins)
  const fft = fftFor(nFft)
  const win = hann(nFft)
  const re = new Float32Array(nFft)
  const im = new Float32Array(nFft)

  for (let f = 0; f < frames; f++) {
    const start = f * hop
    for (let i = 0; i < nFft; i++) {
      re[i] = (samples[start + i] ?? 0) * win[i]
      im[i] = 0
    }
    fft.transform(re, im)
    const base = f * bins
    for (let b = 0; b < bins; b++) mag[base + b] = Math.hypot(re[b], im[b])
  }
  return { mag, frames, bins, hop }
}

// ---------------------------------------------------------------------------
// Mel filterbank (HTK mel scale, normalised triangles)
// ---------------------------------------------------------------------------

const melCache = new Map<string, Float32Array>()

export function melFilterbank(nMels: number, nFft: number, sampleRate: number, fMin = DEFAULTS.fMin, fMax = DEFAULTS.fMax): Float32Array {
  const key = `${nMels}:${nFft}:${sampleRate}:${fMin}:${fMax}`
  const hit = melCache.get(key)
  if (hit) return hit

  const hzToMel = (f: number) => 2595 * Math.log10(1 + f / 700)
  const melToHz = (m: number) => 700 * (Math.pow(10, m / 2595) - 1)

  const bins = nFft / 2 + 1
  const band = new Float32Array(nMels * bins)
  const melMin = hzToMel(fMin)
  const melMax = hzToMel(fMax)
  const edges = new Float32Array(nMels + 2)
  for (let i = 0; i < nMels + 2; i++) {
    edges[i] = melToHz(melMin + ((melMax - melMin) * i) / (nMels + 1))
  }

  for (let m = 0; m < nMels; m++) {
    const lo = edges[m]
    const mid = edges[m + 1]
    const hi = edges[m + 2]
    let sum = 0
    for (let b = 0; b < bins; b++) {
      const hz = (b * sampleRate) / nFft
      let w = 0
      if (hz >= lo && hz <= mid) w = (hz - lo) / (mid - lo)
      else if (hz > mid && hz <= hi) w = (hi - hz) / (hi - mid)
      if (w > 0) {
        band[m * bins + b] = w
        sum += w
      }
    }
    if (sum > 0) for (let b = 0; b < bins; b++) band[m * bins + b] /= sum
  }

  melCache.set(key, band)
  return band
}

/** log-mel spectrogram, nFrames x nMels (row-major, frame-major). */
export function logMel(
  samples: Float32Array,
  sampleRate: number,
  opts: StftOptions = {},
): { data: Float32Array; frames: number; nMels: number; mag: Float32Array; bins: number; hop: number; nFft: number } {
  const nMels = opts.nMels ?? DEFAULTS.nMels
  const nFft = opts.nFft ?? DEFAULTS.nFft
  const { mag, frames, bins, hop } = stftMagnitude(samples, opts)
  const bank = melFilterbank(nMels, nFft, sampleRate, opts.fMin, opts.fMax)
  const data = new Float32Array(frames * nMels)

  for (let f = 0; f < frames; f++) {
    const src = f * bins
    for (let m = 0; m < nMels; m++) {
      let acc = 0
      const base = m * bins
      for (let b = 0; b < bins; b++) acc += mag[src + b] * bank[base + b]
      data[f * nMels + m] = Math.log(acc + 1e-6)
    }
  }
  return { data, frames, nMels, mag, bins, hop, nFft }
}

// ---------------------------------------------------------------------------
// Windowing + pooling into the fixed 64 x 32 model input
// ---------------------------------------------------------------------------

export type Window = { startFrame: number; frames: number }

/** Pick analysis windows of `seconds` length, spread over the clip. */
export function pickWindows(totalFrames: number, hop: number, sampleRate: number, seconds = DEFAULTS.windowSeconds, count = 3): Window[] {
  const frames = Math.max(1, Math.round((seconds * sampleRate) / hop))
  if (totalFrames <= frames) return [{ startFrame: 0, frames: Math.max(1, totalFrames) }]
  const out: Window[] = []
  const usable = totalFrames - frames
  for (let i = 0; i < count; i++) {
    const start = count === 1 ? usable / 2 : (usable * i) / (count - 1)
    out.push({ startFrame: Math.round(start), frames })
  }
  return out
}

/** Frame-major mel window -> pooled 64 x 32 blob laid out band-major (input dim 2048). */
export function poolWindow(
  mel: Float32Array,
  nMels: number,
  window: Window,
  poolFrames = DEFAULTS.poolFrames,
): Float32Array {
  const out = new Float32Array(nMels * poolFrames)
  for (let t = 0; t < poolFrames; t++) {
    const from = window.startFrame + Math.floor((t * window.frames) / poolFrames)
    const to = Math.max(from + 1, window.startFrame + Math.floor(((t + 1) * window.frames) / poolFrames))
    for (let m = 0; m < nMels; m++) {
      let acc = 0
      for (let f = from; f < to; f++) acc += mel[f * nMels + m]
      out[m * poolFrames + t] = acc / (to - from)
    }
  }
  return out
}

export function standardise(x: Float32Array, mean: Float32Array, std: Float32Array, poolFrames = DEFAULTS.poolFrames): Float32Array {
  const out = new Float32Array(x.length)
  for (let i = 0; i < x.length; i++) {
    const band = Math.floor((i % (mean.length * poolFrames)) / poolFrames)
    out[i] = (x[i] - mean[band]) / (std[band] || 1)
  }
  return out
}

/** Full feature extraction for one PCM buffer: 3 windows x 2048 dims. */
export function extractFeatures(
  audio: MonoAudio,
  mean: Float32Array,
  std: Float32Array,
  opts: StftOptions = {},
): Float32Array[] {
  const sr = DEFAULTS.sampleRate
  const samples = resample(audio.samples, audio.sampleRate, sr)
  const mel = logMel(samples, sr, opts)
  const windows = pickWindows(mel.frames, mel.hop, sr)
  const vectorLength = mel.nMels * (opts.poolFrames ?? DEFAULTS.poolFrames)
  const out: Float32Array[] = []
  for (const w of windows) {
    const pooled = poolWindow(mel.data, mel.nMels, w, opts.poolFrames)
    out.push(standardise(pooled, mean, std))
  }
  // pad if the clip is shorter than the window
  while (out.length < 3) out.push(out[out.length - 1] ?? new Float32Array(vectorLength))
  return out
}

// ---------------------------------------------------------------------------
// Pitch contour: the "cartesian" part of the pipeline
// ---------------------------------------------------------------------------

/**
 * Dominant frequency per frame (Hz), plus frame energy (dBFS-ish), which is
 * what the generative step morphs: take the pitch tracks of the clips you
 * liked, blend them, and resynthesise a new call.
 */
export function pitchTrack(
  mag: Float32Array,
  frames: number,
  bins: number,
  sampleRate: number,
  nFft: number,
  hop: number,
): { hz: Float32Array; energy: Float32Array } {
  const hz = new Float32Array(frames)
  const energy = new Float32Array(frames)
  const lowBin = Math.max(1, Math.round((400 * nFft) / sampleRate))
  const highBin = Math.min(bins - 1, Math.round((8000 * nFft) / sampleRate))

  // noise floor per bin: the 20th percentile over time. Subtracting it stops a
  // loud rumble or a constant hum from winning every frame.
  const floor = new Float32Array(bins)
  const column = new Float32Array(frames)
  for (let b = 0; b < bins; b++) {
    for (let f = 0; f < frames; f++) column[f] = mag[f * bins + b]
    column.sort()
    floor[b] = column[Math.floor(frames * 0.2)] ?? 0
  }

  const raw = new Float32Array(frames)
  for (let f = 0; f < frames; f++) {
    let best = 0
    let bestBin = lowBin
    let power = 0
    for (let b = lowBin; b <= highBin; b++) {
      const v = mag[f * bins + b] - floor[b]
      const w = v > 0 ? v : 0
      power += w * w
      if (w > best) {
        best = w
        bestBin = b
      }
    }
    const y0 = Math.max(0, mag[f * bins + Math.max(lowBin, bestBin - 1)] - floor[Math.max(lowBin, bestBin - 1)])
    const y2 = Math.max(0, mag[f * bins + Math.min(highBin, bestBin + 1)] - floor[Math.min(highBin, bestBin + 1)])
    const denom = y0 - 2 * best + y2
    const delta = denom !== 0 ? (0.5 * (y0 - y2)) / denom : 0
    const refined = bestBin + Math.max(-0.5, Math.min(0.5, delta))
    raw[f] = (refined * sampleRate) / nFft
    energy[f] = 20 * Math.log10(Math.sqrt(power / (highBin - lowBin + 1)) + 1e-9)
  }

  // 5-frame median filter: removes single-frame octave jumps without lag
  const half = 2
  const scratch = new Float32Array(2 * half + 1)
  for (let f = 0; f < frames; f++) {
    let n = 0
    for (let k = -half; k <= half; k++) {
      const i = f + k
      if (i >= 0 && i < frames) scratch[n++] = raw[i]
    }
    const view = scratch.subarray(0, n)
    view.sort()
    hz[f] = view[Math.floor(n / 2)]
  }

  return { hz, energy }
}

/** dB envelope -> 0..1 gain per frame */
export function normaliseEnvelope(energy: Float32Array): Float32Array {
  let max = -Infinity
  for (const e of energy) max = Math.max(max, e)
  const out = new Float32Array(energy.length)
  for (let i = 0; i < energy.length; i++) {
    const rel = energy[i] - max
    out[i] = Math.max(0, Math.min(1, Math.pow(10, rel / 20)))
  }
  return out
}
