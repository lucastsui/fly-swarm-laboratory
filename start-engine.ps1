param(
  [ValidateSet('reference', 'experimental')][string]$Model = 'reference',
  [switch]$Run
)
$ErrorActionPreference = 'Stop'
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
  throw 'Create .venv and install dependencies first; see docs/SETUP.md.'
}
Push-Location $PSScriptRoot
try {
  $demoArguments = @('-m', 'engine.run_demo', '--model', $Model)
  if ($Run) { $demoArguments += '--run' }
  & $pythonPath @demoArguments
  if ($LASTEXITCODE -ne 0) { throw "Viewer exited with code $LASTEXITCODE" }
} finally { Pop-Location }
