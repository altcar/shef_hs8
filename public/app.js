// wizzibiz client: swipe deck, taste model readout, chorus synthesis, upload classify.

const $ = (sel) => document.querySelector(sel)

const session = (() => {
  const key = 'wizzibiz-session'
  let id = localStorage.getItem(key)
  if (!id) {
    id = crypto.randomUUID()
    localStorage.setItem(key, id)
  }
  return id
})()

const state = {
  clips: [],
  index: 0,
  swipes: 0,
  likes: 0,
  model: null,
  dsp: null,
  busy: false,
  likedNames: [],
}

const audio = $('#audio')
const card = $('#card')

// ------------------------------------------------------------------ utils ---
const fmt = (n) => (n >= 1000 ? `${(n / 1000).toFixed(1)}k` : `${n}`)

function speciesHue(name) {
  let h = 0
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) % 360
  return h
}

// ------------------------------------------------------------------ deck ----
async function loadDeck() {
  const res = await fetch(`/api/deck?session=${encodeURIComponent(session)}`)
  if (!res.ok) throw new Error(`deck ${res.status}`)
  const data = await res.json()
  state.model = data.model
  state.dsp = data.dsp
  state.clips = data.clips
  state.index = 0
  state.swipes = data.counts.swipes
  state.likes = data.counts.liked

  $('#stat-acc').textContent = `${Math.round(data.model.valAccuracy * 100)}%`
  $('#stat-classes').textContent = fmt(data.model.classes)
  $('#stat-params').textContent = fmt(data.model.params)

  if (data.taste) {
    $('#taste-notready').hidden = true
    $('#taste-ready').hidden = false
    $('#taste-acc').textContent = `${Math.round(data.taste.accuracy * 100)}%`
    $('#taste-likes').textContent = data.counts.liked
  }

  render()
}

function currentClip() {
  return state.clips[state.index] ?? null
}

function render() {
  const clip = currentClip()
  const body = $('#card-body')
  const loading = $('#card-loading')

  if (!clip) {
    loading.hidden = false
    body.hidden = true
    loading.textContent = 'that is the whole forest. reload for another pass, or generate your dawn chorus →'
    return
  }

  loading.hidden = true
  body.hidden = false
  $('#reveal').hidden = true
  $('#pill-hint').textContent = 'tap play, then swipe'
  $('#pill-time').textContent = `${clip.seconds}s · peak ${clip.peakHz} Hz`
  audio.src = `/${clip.file}`
  audio.currentTime = 0
  drawSpectro(clip)
  drawMap(clip)

  card.classList.remove('swipe-like', 'swipe-nope')
  card.classList.add('enter')
  setTimeout(() => card.classList.remove('enter'), 340)

  $('#bar').style.width = `${Math.min(100, (state.swipes / 20) * 100)}%`
}

// ------------------------------------------------------------- spectrogram --
function drawSpectro(clip) {
  const canvas = $('#spectro')
  const ctx = canvas.getContext('2d')
  const { width: W, height: H } = canvas
  ctx.clearRect(0, 0, W, H)

  // background
  const bg = ctx.createLinearGradient(0, 0, 0, H)
  bg.addColorStop(0, 'rgba(255,207,122,0.10)')
  bg.addColorStop(1, 'rgba(90,40,80,0.05)')
  ctx.fillStyle = bg
  ctx.fillRect(0, 0, W, H)

  // frequency grid
  ctx.strokeStyle = 'rgba(255,255,255,0.06)'
  ctx.fillStyle = 'rgba(255,255,255,0.35)'
  ctx.font = '11px ui-monospace, monospace'
  ctx.lineWidth = 1
  const maxHz = 8000
  for (let hz = 1000; hz <= maxHz; hz += 1000) {
    const y = H - (hz / maxHz) * (H - 20) - 10
    ctx.beginPath()
    ctx.moveTo(0, y)
    ctx.lineTo(W, y)
    ctx.stroke()
    ctx.fillText(`${hz / 1000}k`, 6, y - 4)
  }
  // time grid
  for (let s = 1; s < 5; s++) {
    const x = (s / 5) * W
    ctx.beginPath()
    ctx.moveTo(x, 0)
    ctx.lineTo(x, H)
    ctx.stroke()
  }

  const contour = clip.contour ?? []
  if (!contour.length) return
  const toY = (hz) => H - (Math.min(maxHz, Math.max(0, hz)) / maxHz) * (H - 20) - 10
  const toX = (i) => (i / (contour.length - 1)) * W

  // envelope underneath
  ctx.beginPath()
  ctx.moveTo(0, H)
  contour.forEach((_, i) => ctx.lineTo(toX(i), H - 6))
  ctx.lineTo(W, H)
  ctx.closePath()
  ctx.fillStyle = 'rgba(255,207,122,0.10)'
  ctx.fill()

  // pitch track
  ctx.beginPath()
  contour.forEach((hz, i) => (i ? ctx.lineTo(toX(i), toY(hz)) : ctx.moveTo(toX(i), toY(hz))))
  ctx.strokeStyle = '#ffcf7a'
  ctx.lineWidth = 2.4
  ctx.lineJoin = 'round'
  ctx.shadowColor = 'rgba(255,207,122,0.55)'
  ctx.shadowBlur = 12
  ctx.stroke()
  ctx.shadowBlur = 0

  // glow points where the bird is loudest
  contour.forEach((hz, i) => {
    const e = clip.envelope?.[i] ?? 0
    if (e < 0.35) return
    ctx.beginPath()
    ctx.arc(toX(i), toY(hz), 2 + e * 3, 0, Math.PI * 2)
    ctx.fillStyle = `rgba(255,255,255,${0.25 + e * 0.5})`
    ctx.fill()
  })
}

// ------------------------------------------------------------------ map -----
let mapPoints = []
function drawMap(active) {
  const canvas = $('#map')
  const ctx = canvas.getContext('2d')
  const { width: W, height: H } = canvas
  ctx.clearRect(0, 0, W, H)

  const all = state.clips.filter((c) => c.pca)
  if (!all.length) return
  const xs = all.map((c) => c.pca[0])
  const ys = all.map((c) => c.pca[1])
  const minX = Math.min(...xs), maxX = Math.max(...xs)
  const minY = Math.min(...ys), maxY = Math.max(...ys)
  const px = (x) => 14 + ((x - minX) / (maxX - minX || 1)) * (W - 28)
  const py = (y) => H - 14 - ((y - minY) / (maxY - minY || 1)) * (H - 28)

  for (const clip of all) {
    const isActive = active && clip.id === active.id
    ctx.beginPath()
    ctx.arc(px(clip.pca[0]), py(clip.pca[1]), isActive ? 6 : 4, 0, Math.PI * 2)
    ctx.fillStyle = `hsla(${speciesHue(clip.name)}, 70%, ${isActive ? 78 : 60}%, ${isActive ? 1 : 0.65})`
    if (isActive) {
      ctx.shadowColor = 'rgba(255,255,255,0.8)'
      ctx.shadowBlur = 14
    }
    ctx.fill()
    ctx.shadowBlur = 0
    if (isActive) {
      ctx.strokeStyle = 'rgba(255,255,255,0.9)'
      ctx.lineWidth = 1.5
      ctx.stroke()
    }
  }
}

// ---------------------------------------------------------------- swipe -----
async function swipe(liked) {
  if (state.busy) return
  const clip = currentClip()
  if (!clip) return
  state.busy = true

  card.classList.add(liked ? 'swipe-like' : 'swipe-nope')
  audio.pause()

  try {
    await fetch('/api/swipe', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ session, clip: clip.id, liked }),
    })
  } catch (err) {
    console.warn('swipe failed', err)
  }

  state.swipes++
  if (liked) {
    state.likes++
    state.likedNames.push(clip.name)
  }

  // reveal the truth
  const top = clip.top?.[0]
  $('#reveal').hidden = false
  $('#reveal-name').textContent = top ? `${top.name}` : clip.name
  $('#reveal-sub').textContent = `${Math.round((top?.p ?? 0) * 100)}% confident · ${clip.name === top?.name ? 'the model got it right' : `actually ${clip.name}`}`
  $('#pill-hint').textContent = liked ? 'liked' : 'skipped'

  setTimeout(() => {
    state.index++
    state.busy = false
    render()
    if (state.swipes % 4 === 0) refreshTaste()
  }, 330)
}

async function refreshTaste() {
  try {
    const res = await fetch(`/api/taste?session=${encodeURIComponent(session)}`)
    const data = await res.json()
    if (!data.ready) return
    $('#taste-notready').hidden = true
    $('#taste-ready').hidden = false
    $('#taste-acc').textContent = `${Math.round(data.accuracy * 100)}%`
    $('#taste-likes').textContent = data.liked
    const list = $('#taste-species')
    list.innerHTML = ''
    for (const s of data.topSpecies) {
      const li = document.createElement('li')
      li.innerHTML = `<span>${s.name}</span><span>${s.count}×</span>`
      list.appendChild(li)
    }
    window.__taste = data
  } catch (err) {
    console.warn('taste failed', err)
  }
}

// --------------------------------------------------------- chorus synth -----
async function generateChorus() {
  const data = window.__taste
  if (!data?.contour) {
    await refreshTaste()
  }
  const taste = window.__taste
  if (!taste?.contour) {
    alert('like at least one bird first 🐦')
    return
  }

  const ctx = new (window.AudioContext || window.webkitAudioContext)()
  const master = ctx.createGain()
  master.gain.value = 0.9
  // gentle lowpass keeps the harmonic stack without gutting low calls
  const filter = ctx.createBiquadFilter()
  filter.type = 'lowpass'
  filter.frequency.value = 9000
  filter.Q.value = 0.6
  filter.connect(master).connect(ctx.destination)

  const highpass = ctx.createBiquadFilter()
  highpass.type = 'highpass'
  highpass.frequency.value = 180
  highpass.connect(filter)

  const { hz, env } = taste.contour
  const dt = 5 / hz.length

  hz.forEach((f, i) => {
    const amp = env?.[i] ?? 0.5
    if (!f || amp < 0.06) return
    const t = ctx.currentTime + i * dt

    for (const [mult, gain] of [
      [1, 1],
      [2, 0.28],
      [3, 0.1],
    ]) {
      const osc = ctx.createOscillator()
      osc.type = 'sine'
      osc.frequency.setValueAtTime(f * mult, t)
      const next = hz[i + 1] ?? f
      osc.frequency.linearRampToValueAtTime(next * mult, t + dt)

      const g = ctx.createGain()
      g.gain.setValueAtTime(0, t)
      g.gain.linearRampToValueAtTime(0.18 * gain * (0.4 + amp), t + dt * 0.25)
      g.gain.exponentialRampToValueAtTime(0.0001, t + dt * 0.95)

      osc.connect(g).connect(highpass)
      osc.start(t)
      osc.stop(t + dt)
    }
  })

  // a whisper of dawn air behind it
  const noiseBuffer = ctx.createBuffer(1, ctx.sampleRate * 5, ctx.sampleRate)
  const noise = noiseBuffer.getChannelData(0)
  for (let i = 0; i < noise.length; i++) noise[i] = (Math.random() * 2 - 1) * 0.012
  const noiseSrc = ctx.createBufferSource()
  noiseSrc.buffer = noiseBuffer
  const noiseFilter = ctx.createBiquadFilter()
  noiseFilter.type = 'highpass'
  noiseFilter.frequency.value = 3000
  const noiseGain = ctx.createGain()
  noiseGain.gain.value = 0.35
  noiseSrc.connect(noiseFilter).connect(noiseGain).connect(ctx.destination)
  noiseSrc.start(ctx.currentTime)

  setTimeout(() => ctx.close(), 6000)
}

// -------------------------------------------------------------- classify ----
function encodeWav16(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2)
  const view = new DataView(buffer)
  const writeStr = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i))
  }
  writeStr(0, 'RIFF')
  view.setUint32(4, 36 + samples.length * 2, true)
  writeStr(8, 'WAVEfmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, 1, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  writeStr(36, 'data')
  view.setUint32(40, samples.length * 2, true)
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true)
  }
  return new Uint8Array(buffer)
}

async function classifyFile(file) {
  const out = $('#upload-result')
  out.textContent = 'decoding…'
  try {
    const ctx = new AudioContext()
    const arrayBuffer = await file.arrayBuffer()
    const decoded = await ctx.decodeAudioData(arrayBuffer)
    // resample to 16 kHz mono exactly like the training pipeline
    const targetRate = 16000
    const length = Math.max(1, Math.floor((decoded.length / decoded.sampleRate) * targetRate))
    const offline = new OfflineAudioContext(1, length, targetRate)
    const src = offline.createBufferSource()
    const mono = offline.createBuffer(1, decoded.length, decoded.sampleRate)
    const merged = mono.getChannelData(0)
    for (let c = 0; c < decoded.numberOfChannels; c++) {
      const ch = decoded.getChannelData(c)
      for (let i = 0; i < ch.length; i++) merged[i] += ch[i] / decoded.numberOfChannels
    }
    src.buffer = mono
    src.connect(offline.destination)
    src.start()
    const rendered = await offline.startRendering()
    await ctx.close()

    out.textContent = 'analysing on the Worker…'
    const res = await fetch('/api/classify', { method: 'POST', body: encodeWav16(rendered.getChannelData(0), targetRate) })
    const data = await res.json()
    if (!res.ok) throw new Error(data.error ?? res.status)

    out.innerHTML = ''
    for (const t of data.top) {
      const row = document.createElement('div')
      row.className = 'bar-row'
      row.innerHTML = `<span>${t.code.replace(/_/g, ' ')}</span><span>${Math.round(t.p * 100)}%</span>
        <div class="meter"><i style="width:${Math.round(t.p * 100)}%"></i></div>`
      out.appendChild(row)
    }
    const note = document.createElement('div')
    note.className = 'muted small'
    note.textContent = data.notABird > 0.5 ? 'the model thinks this is probably not a bird.' : 'sounds like a bird to me.'
    out.appendChild(note)
  } catch (err) {
    out.textContent = `could not analyse that file: ${err.message}`
  }
}

// ---------------------------------------------------------------- wiring ----
$('#like').addEventListener('click', () => swipe(true))
$('#nope').addEventListener('click', () => swipe(false))
$('#generate').addEventListener('click', generateChorus)
$('#play').addEventListener('click', async () => {
  if (audio.paused) {
    await audio.play()
    $('#play').classList.add('playing')
  } else {
    audio.pause()
  }
})
audio.addEventListener('ended', () => $('#play').classList.remove('playing'))
audio.addEventListener('pause', () => $('#play').classList.remove('playing'))

$('#file').addEventListener('change', (e) => {
  const file = e.target.files?.[0]
  if (file) classifyFile(file)
})

window.addEventListener('keydown', (e) => {
  if (e.target instanceof HTMLInputElement) return
  if (e.key === 'ArrowRight') swipe(true)
  if (e.key === 'ArrowLeft') swipe(false)
  if (e.key === ' ') {
    e.preventDefault()
    $('#play').click()
  }
})

// drag to swipe
let drag = null
card.addEventListener('pointerdown', (e) => {
  drag = { x: e.clientX, y: e.clientY }
  card.setPointerCapture(e.pointerId)
})
card.addEventListener('pointermove', (e) => {
  if (!drag) return
  const dx = e.clientX - drag.x
  const dy = e.clientY - drag.y
  if (Math.abs(dx) < 4) return
  card.style.transform = `translate(${dx}px, ${dy * 0.25}px) rotate(${dx * 0.03}deg)`
  card.style.opacity = `${Math.max(0.4, 1 - Math.abs(dx) / 700)}`
})
card.addEventListener('pointerup', (e) => {
  if (!drag) return
  const dx = e.clientX - drag.x
  drag = null
  card.style.transform = ''
  card.style.opacity = ''
  if (Math.abs(dx) > 110) swipe(dx > 0)
})

loadDeck().then(refreshTaste).catch((err) => {
  $('#card-loading').textContent = `could not load the deck: ${err.message}`
})
