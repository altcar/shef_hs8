$ErrorActionPreference = 'Stop'
$base = 'https://wizzibiz.hheylau.workers.dev'
$root = 'C:\Users\c85-lau\AppData\Local\Temp\opencode\birds\wizzibiz'
$deck = (Invoke-RestMethod "$base/api/deck?session=consistency").clips
$sample = $deck | Get-Random -Count 12

$ok = 0
$agree = 0
foreach ($clip in $sample) {
  $bytes = [System.IO.File]::ReadAllBytes("$root\public\clips\$($clip.id).wav")
  $cls = Invoke-RestMethod "$base/api/classify" -Method Post -Body $bytes -ContentType 'audio/wav'
  $match = $cls.top[0].code -eq $clip.top[0].code
  $correct = $cls.top[0].code -eq $clip.label
  if ($match) { $agree++ }
  if ($correct) { $ok++ }
  "{0,-10} {1,-28} worker={2,-9} deck={3,-9} {4}" -f $clip.id, $clip.name, $cls.top[0].code, $clip.top[0].code, $(if ($match) { 'match' } else { 'MISMATCH' })
}
""
"worker/export agreement: $agree/$($sample.Count)"
"top-1 correct on this sample: $ok/$($sample.Count)"
