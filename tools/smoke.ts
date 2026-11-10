/**
 * smoke.ts - checks the quantised model end to end on a real clip.
 * Usage: node tools/smoke.ts public/clips/bird-0.wav
 */
import { readFileSync, readdirSync } from 'node:fs'
import { decodeWav, extractFeatures } from '../src/dsp.ts'
import { loadModel, predictWindows } from '../src/model.ts'
import { MODEL_DATA, STATS } from '../src/model-data.ts'

const model = loadModel(MODEL_DATA)
const stats = { mean: Float32Array.from(STATS.mean), std: Float32Array.from(STATS.std) }

const files = process.argv[2]
  ? [process.argv[2]]
  : readdirSync('C:/Users/c85-lau/AppData/Local/Temp/opencode/birds/wizzibiz/public/clips')
      .filter((f) => f.endsWith('.wav'))
      .slice(0, 6)
      .map((f) => `C:/Users/c85-lau/AppData/Local/Temp/opencode/birds/wizzibiz/public/clips/${f}`)

console.log(`model: ${MODEL_DATA.spec.map((s) => s.type).join(' -> ')}`)
console.log(`classes: ${model.classes.length}, input ${model.inputH}x${model.inputW}\n`)

for (const file of files) {
  const bytes = new Uint8Array(readFileSync(file))
  const audio = decodeWav(bytes)
  const t0 = performance.now()
  const features = extractFeatures(audio, stats.mean, stats.std)
  const pred = predictWindows(model, features)
  const ms = performance.now() - t0
  const top = pred.top
    .map((t) => `${model.classes[t.index]}=${(t.p * 100).toFixed(0)}%`)
    .join('  ')
  console.log(`${file.split(/[\\/]/).pop()?.padEnd(22)} ${top}   (${ms.toFixed(1)}ms)`)
  if (!Number.isFinite(pred.probs[0])) throw new Error('non-finite probabilities')
}
