/**
 * fetch-data.mjs - downloads the two parquet shards used for training.
 * ~1.3 GB total, written to DATA_DIR (override with WIZZIBIZ_DATA).
 *
 *   node tools/fetch-data.mjs
 */
import { createWriteStream, mkdirSync } from 'node:fs'
import { Readable } from 'node:stream'
import { pipeline } from 'node:stream/promises'

const DIR = process.env.WIZZIBIZ_DATA ?? 'C:/Users/c85-lau/AppData/Local/Temp/opencode/birds/data'

const FILES = [
  ['https://huggingface.co/datasets/tglcourse/5s_birdcall_samples_top20/resolve/main/data/train-00000-of-00006.parquet', 'birds-00000.parquet'],
  ['https://huggingface.co/datasets/tglcourse/5s_birdcall_samples_top20/resolve/main/data/train-00001-of-00006.parquet', 'birds-00001.parquet'],
  ['https://huggingface.co/datasets/ashraq/esc50/resolve/main/data/train-00000-of-00002-2f1ab7b824ec751f.parquet', 'esc50-00000.parquet'],
]

mkdirSync(DIR, { recursive: true })

for (const [url, name] of FILES) {
  const target = `${DIR}/${name}`
  process.stdout.write(`downloading ${name} ... `)
  const res = await fetch(url)
  if (!res.ok || !res.body) throw new Error(`${name}: HTTP ${res.status}`)
  await pipeline(Readable.fromWeb(res.body), createWriteStream(target))
  console.log('done')
}

console.log(`\nok -> ${DIR}`)
