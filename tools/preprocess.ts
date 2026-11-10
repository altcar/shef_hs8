/**
 * preprocess.ts
 *
 * Reads the parquet shards (bird clips from xeno-canto + ESC-50 negatives),
 * decodes the embedded WAV bytes, runs the DSP pipeline from src/dsp.ts and
 * writes a compact feature store for training.
 *
 * Usage: node tools/preprocess.ts [--limit N]
 */
import { asyncBufferFromFile, parquetMetadataAsync, parquetReadObjects } from 'hyparquet'
import { mkdir, writeFile } from 'node:fs/promises'
import {
  DEFAULTS,
  FEATURE_DIM,
  decodeWav,
  logMel,
  pitchTrack,
  poolWindow,
  pickWindows,
  normaliseEnvelope,
  resample,
} from '../src/dsp.ts'

const DATA_DIR = 'C:/Users/c85-lau/AppData/Local/Temp/opencode/birds/data'
const CACHE = 'C:/Users/c85-lau/AppData/Local/Temp/opencode/birds/wizzibiz/cache'
const WINDOWS = 3
const CONTOUR_POINTS = 50

const limitArg = process.argv.indexOf('--limit')
const LIMIT = limitArg > -1 ? Number(process.argv[limitArg + 1]) : Infinity

type Clip = {
  id: string
  source: string
  row: number
  label: string
  seconds: number
  rms: number
  peakHz: number
  contour: number[]
  envelope: number[]
}

type Shard = { path: string; source: string; labelMode: 'label' | 'esc50'; take: number }

const shards: Shard[] = [
  { path: `${DATA_DIR}/birds-00000.parquet`, source: 'bird', labelMode: 'label', take: Infinity },
  { path: `${DATA_DIR}/birds-00001.parquet`, source: 'bird', labelMode: 'label', take: Infinity },
  { path: `${DATA_DIR}/esc50-00000.parquet`, source: 'esc50', labelMode: 'esc50', take: Infinity },
]

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

function toMono16k(samples: Float32Array, sampleRate: number): Float32Array {
  return sampleRate === DEFAULTS.sampleRate
    ? samples
    : resample(samples, sampleRate, DEFAULTS.sampleRate)
}

await mkdir(CACHE, { recursive: true })

const clips: Clip[] = []
const featureChunks: Float32Array[] = []
let skipped = 0
const started = Date.now()

for (const shard of shards) {
  const file = await asyncBufferFromFile(shard.path)
  const meta = await parquetMetadataAsync(file)
  const totalRows = Number(meta.num_rows)
  console.log(`\nreading ${shard.path} (${totalRows} rows)...`)
  const t0 = Date.now()
  const rows = await parquetReadObjects({
    file,
    columns: shard.labelMode === 'esc50' ? ['audio', 'category'] : ['audio', 'label'],
    utf8: false,
  })
  console.log(`  read in ${((Date.now() - t0) / 1000).toFixed(1)}s, heap ${Math.round(process.memoryUsage().heapUsed / 1e6)}MB`)

  let done = 0
  for (let k = 0; k < rows.length && done < LIMIT; k++) {
    const row = rows[k]
    // release the audio bytes as we go so a 480 MB shard does not stay resident
    rows[k] = null as unknown as (typeof rows)[number]
    const audioField = row.audio as { bytes?: Uint8Array } | undefined
    if (!audioField?.bytes) {
      skipped++
      continue
    }
      const label =
        shard.labelMode === 'esc50'
          ? String(row.category) === 'chirping_birds'
            ? null // keep the negative class clean: skip generic bird chirps
            : 'not_a_bird'
          : String(row.label)
      if (!label) {
        skipped++
        continue
      }

      let decoded
      try {
        decoded = decodeWav(audioField.bytes)
      } catch {
        skipped++
        continue
      }

      const samples = toMono16k(decoded.samples, decoded.sampleRate)
      const mel = logMel(samples, DEFAULTS.sampleRate)
      if (mel.frames < 20) {
        skipped++
        continue
      }

      const windows = pickWindows(mel.frames, mel.hop, DEFAULTS.sampleRate, DEFAULTS.windowSeconds, WINDOWS)
      const block = new Float32Array(WINDOWS * FEATURE_DIM)
      for (let w = 0; w < WINDOWS; w++) {
        const win = windows[Math.min(w, windows.length - 1)]
        block.set(poolWindow(mel.data, mel.nMels, win, DEFAULTS.poolFrames), w * FEATURE_DIM)
      }
      featureChunks.push(block)

      const track = pitchTrack(mel.mag, mel.frames, mel.bins, DEFAULTS.sampleRate, mel.nFft, mel.hop)
      const env = normaliseEnvelope(track.energy)

      let rms = 0
      for (let i = 0; i < samples.length; i++) rms += samples[i] * samples[i]
      rms = Math.sqrt(rms / Math.max(1, samples.length))

      const contour = downsample(track.hz, CONTOUR_POINTS)
      const sorted = [...contour].sort((a, b) => a - b)
      clips.push({
        id: `${shard.source}-${k}`,
        source: shard.source,
        row: k,
        label,
        seconds: Math.round((samples.length / DEFAULTS.sampleRate) * 100) / 100,
        rms: Math.round(rms * 1000) / 1000,
        peakHz: sorted[Math.floor(sorted.length * 0.9)] ?? 0,
        contour,
        envelope: downsample(env, CONTOUR_POINTS).map((v) => Math.round(v * 100) / 100),
      })
      done++

      if (clips.length % 200 === 0) {
        const perSec = clips.length / ((Date.now() - started) / 1000)
        console.log(`  ${clips.length} clips (${perSec.toFixed(1)}/s) last=${label} skipped=${skipped}`)
      }
  }
  console.log(`shard done: ${shard.path} -> ${clips.length} total clips`)
}

// --- write feature blob -----------------------------------------------------
const totalFloats = clips.length * WINDOWS * FEATURE_DIM
const flat = new Float32Array(totalFloats)
let offset = 0
for (const chunk of featureChunks) {
  flat.set(chunk, offset)
  offset += chunk.length
}
await writeFile(`${CACHE}/raw.f32`, Buffer.from(flat.buffer, flat.byteOffset, flat.byteLength))

// --- per-band statistics ----------------------------------------------------
const mean = new Float64Array(DEFAULTS.nMels)
const sq = new Float64Array(DEFAULTS.nMels)
let count = 0
for (const block of featureChunks) {
  for (let w = 0; w < WINDOWS; w++) {
    const base = w * FEATURE_DIM
    for (let m = 0; m < DEFAULTS.nMels; m++) {
      for (let t = 0; t < DEFAULTS.poolFrames; t++) {
        const v = block[base + m * DEFAULTS.poolFrames + t]
        mean[m] += v
        sq[m] += v * v
        count++
      }
    }
  }
}
const meanArr = new Array(DEFAULTS.nMels).fill(0)
const stdArr = new Array(DEFAULTS.nMels).fill(1)
for (let m = 0; m < DEFAULTS.nMels; m++) {
  const mu = mean[m] / (count / DEFAULTS.nMels)
  const variance = sq[m] / (count / DEFAULTS.nMels) - mu * mu
  meanArr[m] = Math.round(mu * 1e4) / 1e4
  stdArr[m] = Math.round(Math.max(1e-3, Math.sqrt(Math.max(0, variance))) * 1e4) / 1e4
}

await writeFile(`${CACHE}/stats.json`, JSON.stringify({ mean: meanArr, std: stdArr, windows: WINDOWS, dim: FEATURE_DIM }))
await writeFile(`${CACHE}/meta.json`, JSON.stringify({ windows: WINDOWS, dim: FEATURE_DIM, clips }, null, 0))

console.log(
  `\nwrote ${clips.length} clips (${(totalFloats * 4 / 1e6).toFixed(1)} MB features) in ${((Date.now() - started) / 1000).toFixed(0)}s, skipped=${skipped}`,
)
