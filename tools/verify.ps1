$ErrorActionPreference = 'Stop'
$base = 'https://wizzibiz.hheylau.workers.dev'
$session = 'verify-' + [guid]::NewGuid().ToString('N').Substring(0, 8)

function Show($label, $value) { "{0,-22} {1}" -f $label, $value }

# 1. static site
$r = Invoke-WebRequest "$base/" -UseBasicParsing
Show 'GET /' "$($r.StatusCode) $($r.Headers['content-type']) $([math]::Round($r.RawContentLength/1kb,1))kB"

# 2. deck
$deck = Invoke-RestMethod "$base/api/deck?session=$session"
Show 'GET /api/deck' "$($deck.clips.Count) clips, model acc $($deck.model.valAccuracy), $($deck.model.params) params"
Show '  first clip' "$($deck.clips[0].id) $($deck.clips[0].name) top=$($deck.clips[0].top[0].code) $([math]::Round($deck.clips[0].top[0].p*100))%"

# 3. audio asset
$wav = Invoke-WebRequest "$base/$($deck.clips[0].file)" -UseBasicParsing
Show 'GET clip wav' "$($wav.StatusCode) $($wav.Headers['content-type']) $([math]::Round($wav.RawContentLength/1kb,1))kB"

# 4. swipes
$picks = $deck.clips | Select-Object -First 8
for ($i = 0; $i -lt $picks.Count; $i++) {
  $body = @{ session = $session; clip = $picks[$i].id; liked = ($i % 2 -eq 0) } | ConvertTo-Json
  Invoke-RestMethod "$base/api/swipe" -Method Post -Body $body -ContentType 'application/json' | Out-Null
}
Show 'POST /api/swipe' "8 swipes written to D1"

# 5. taste model
$taste = Invoke-RestMethod "$base/api/taste?session=$session"
Show 'GET /api/taste' "ready=$($taste.ready) acc=$($taste.accuracy) liked=$($taste.liked)/$($taste.swipes)"
Show '  top species' (($taste.topSpecies | ForEach-Object { "$($_.name) x$($_.count)" }) -join ', ')
Show '  contour' "$($taste.contour.hz.Count) points, $($taste.contour.hz[0]) -> $($taste.contour.hz[-1]) Hz"

# 6. re-ranked deck excludes swiped clips
$deck2 = Invoke-RestMethod "$base/api/deck?session=$session"
$seen = ($deck2.clips | Where-Object { $picks.id -contains $_.id }).Count
Show 'deck after swipes' "$($deck2.clips.Count) left, $seen of the 8 swiped still present, taste acc $($deck2.taste.accuracy)"

# 7. server-side classification of a real clip, and consistency with the
#    prediction baked into clips.json at export time
$target = $deck.clips[0]
$bytes = [System.IO.File]::ReadAllBytes("C:\Users\c85-lau\AppData\Local\Temp\opencode\birds\wizzibiz\public\clips\$($target.id).wav")
$cls = Invoke-RestMethod "$base/api/classify" -Method Post -Body $bytes -ContentType 'audio/wav'
Show 'POST /api/classify' (($cls.top | ForEach-Object { "$($_.code)=$([math]::Round($_.p*100))%" }) -join '  ')
Show '  not_a_bird' $cls.notABird
Show '  expect (deck)' "$($target.top[0].code)=$([math]::Round($target.top[0].p*100))% for $($target.id) ($($target.name))"
Show '  consistent' ($cls.top[0].code -eq $target.top[0].code)

# 8. 404 behaviour
try { Invoke-RestMethod "$base/api/nope" | Out-Null } catch { Show 'GET /api/nope' "$($_.Exception.Response.StatusCode.value__)" }
