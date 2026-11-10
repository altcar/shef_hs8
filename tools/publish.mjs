/**
 * publish.mjs - creates altcar/wizzibiz (if missing) and uploads the project
 * to it using the GitHub Git Data API. Works without a git binary.
 *
 *   node tools/publish.mjs [--message "..."]
 */
import { execFileSync } from 'node:child_process'
import { readFileSync, readdirSync, statSync } from 'node:fs'
import { join, relative, sep } from 'node:path'

const ROOT = 'C:/Users/c85-lau/AppData/Local/Temp/opencode/birds/wizzibiz'
const OWNER = 'altcar'
const REPO = 'hs8'
const BRANCH = 'main'
const SKIP = new Set(['node_modules', 'cache', '.wrangler', '.git', '.dev.vars'])
const SKIP_EXT = ['.log']

const messageIdx = process.argv.indexOf('--message')
const message = messageIdx > -1 ? process.argv[messageIdx + 1] : 'wizzibiz: build + deploy'

const token = execFileSync('gh', ['auth', 'token'], { encoding: 'utf8' }).trim()

async function api(path, init = {}) {
  const res = await fetch(`https://api.github.com${path}`, {
    ...init,
    headers: {
      authorization: `Bearer ${token}`,
      accept: 'application/vnd.github+json',
      'content-type': 'application/json',
      'user-agent': 'wizzibiz-publish',
      ...(init.headers ?? {}),
    },
  })
  const text = await res.text()
  const body = text ? JSON.parse(text) : null
  if (!res.ok) throw new Error(`${init.method ?? 'GET'} ${path} -> ${res.status} ${text.slice(0, 300)}`)
  return body
}

// ---- repo ------------------------------------------------------------------
let repo
try {
  repo = await api(`/repos/${OWNER}/${REPO}`)
  console.log(`repo exists: ${repo.full_name} (id ${repo.id})`)
} catch (error) {
  if (!String(error).includes('404')) throw error
  repo = await api('/user/repos', {
    method: 'POST',
    body: JSON.stringify({
      name: REPO,
      description: 'Swipe on the dawn chorus. DSP + a small CNN on Cloudflare Workers.',
      homepage: 'https://wizzibiz.hheylau.workers.dev',
      private: false,
      has_issues: true,
      has_wiki: false,
    }),
  })
  console.log(`created repo: ${repo.full_name} (id ${repo.id})`)
}

// ---- collect files ---------------------------------------------------------
// GitHub needs at least one commit before blob/tree APIs work, so bootstrap
// an empty repository with a placeholder file.
let existingRef = null
try {
  existingRef = await api(`/repos/${OWNER}/${REPO}/git/ref/heads/${BRANCH}`)
} catch {
  existingRef = null
}
if (!existingRef) {
  console.log('bootstrapping empty repository...')
  await api(`/repos/${OWNER}/${REPO}/contents/.bootstrap`, {
    method: 'PUT',
    body: JSON.stringify({ message: 'chore: bootstrap', content: Buffer.from('wizzibiz\n').toString('base64') }),
  })
}

function walk(dir, out = []) {
  for (const entry of readdirSync(dir)) {
    if (SKIP.has(entry)) continue
    const full = join(dir, entry)
    const stat = statSync(full)
    if (stat.isDirectory()) walk(full, out)
    else {
      if (SKIP_EXT.some((ext) => entry.endsWith(ext))) continue
      out.push(full)
    }
  }
  return out
}

const files = walk(ROOT)
console.log(`uploading ${files.length} files...`)

const blobs = []
let done = 0
const queue = [...files]
async function worker() {
  for (;;) {
    const file = queue.shift()
    if (!file) return
    const content = readFileSync(file).toString('base64')
    const blob = await api(`/repos/${OWNER}/${REPO}/git/blobs`, {
      method: 'POST',
      body: JSON.stringify({ content, encoding: 'base64' }),
    })
    blobs.push({ path: relative(ROOT, file).split(sep).join('/'), mode: '100644', type: 'blob', sha: blob.sha })
    done++
    if (done % 20 === 0) console.log(`  ${done}/${files.length}`)
  }
}
await Promise.all([worker(), worker(), worker(), worker(), worker(), worker()])

const tree = await api(`/repos/${OWNER}/${REPO}/git/trees`, {
  method: 'POST',
  body: JSON.stringify({ tree: blobs }),
})
console.log(`tree ${tree.sha} with ${blobs.length} entries`)

// keep history linear: parent = current branch head if it exists
let parents = []
try {
  const ref = await api(`/repos/${OWNER}/${REPO}/git/ref/heads/${BRANCH}`)
  parents = [ref.object.sha]
} catch {
  console.log('first push (no existing ref)')
}
const commit = await api(`/repos/${OWNER}/${REPO}/git/commits`, {
  method: 'POST',
  body: JSON.stringify({ message, tree: tree.sha, parents }),
})

if (parents.length) {
  await api(`/repos/${OWNER}/${REPO}/git/refs/heads/${BRANCH}`, {
    method: 'PATCH',
    body: JSON.stringify({ sha: commit.sha, force: true }),
  })
} else {
  await api(`/repos/${OWNER}/${REPO}/git/refs`, {
    method: 'POST',
    body: JSON.stringify({ ref: `refs/heads/${BRANCH}`, sha: commit.sha }),
  })
}

console.log(`pushed ${commit.sha.slice(0, 7)} -> https://github.com/${OWNER}/${REPO}`)

