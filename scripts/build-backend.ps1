$ErrorActionPreference = 'Stop'

Write-Host 'Building React UI (static assets)...'
Push-Location 'apps\dsa-web'
if (!(Test-Path 'node_modules')) {
  npm install
}
npm run build
Pop-Location

$pythonBin = $env:PYTHON_BIN
if ([string]::IsNullOrWhiteSpace($pythonBin)) {
  $pythonBin = 'python'
}

Write-Host "Using Python: $pythonBin"

function Test-PythonCode {
  param(
    [string]$Python,
    [string]$Code
  )

  try {
    & $Python -c $Code *> $null
    return ($LASTEXITCODE -eq 0)
  } catch {
    return $false
  }
}

Write-Host 'Building backend executable...'
if (-not (Test-PythonCode -Python $pythonBin -Code "import PyInstaller")) {
  & $pythonBin -m pip install pyinstaller
}

Write-Host 'Installing backend dependencies...'
& $pythonBin -m pip install -r requirements.txt

Write-Host 'Checking python-multipart availability...'
if (-not (Test-PythonCode -Python $pythonBin -Code "import multipart, multipart.multipart")) {
  throw 'python-multipart is not importable in the selected Python environment.'
}

if (Test-Path 'dist\backend') {
  Remove-Item -Recurse -Force 'dist\backend'
}
New-Item -ItemType Directory -Path 'dist\backend' | Out-Null

if (Test-Path 'dist\stock_analysis') {
  Remove-Item -Recurse -Force 'dist\stock_analysis'
}

if (Test-Path 'build\stock_analysis') {
  Remove-Item -Recurse -Force 'build\stock_analysis'
}

$hiddenImports = @(
  'multipart',
  'multipart.multipart',
  'json_repair',
  'api',
  'api.app',
  'api.deps',
  'api.v1',
  'api.v1.router',
  'api.v1.endpoints',
  'api.v1.endpoints.analysis',
  'api.v1.endpoints.history',
  'api.v1.endpoints.stocks',
  'api.v1.endpoints.health',
  'api.v1.schemas',
  'api.v1.schemas.analysis',
  'api.v1.schemas.history',
  'api.v1.schemas.stocks',
  'api.v1.schemas.common',
  'api.middlewares',
  'api.middlewares.error_handler',
  'src.services',
  'src.services.task_queue',
  'src.services.analysis_service',
  'src.services.history_service',
  'uvicorn.logging',
  'uvicorn.loops',
  'uvicorn.loops.auto',
  'uvicorn.protocols',
  'uvicorn.protocols.http',
  'uvicorn.protocols.http.auto',
  'uvicorn.protocols.websockets',
  'uvicorn.protocols.websockets.auto',
  'uvicorn.lifespan',
  'uvicorn.lifespan.on'
)
$hiddenImportArgs = $hiddenImports | ForEach-Object { "--hidden-import=$_" }

# 设置 matplotlib 后端为非交互式，避免 PyInstaller 打包 Qt 相关依赖
$env:MPLBACKEND = 'Agg'

$pyInstallerArgs = @(
  '-m', 'PyInstaller',
  '--name', 'stock_analysis',
  '--onedir',
  '--noconfirm',
  '--noconsole',
  '--add-data', 'static;static',
  '--collect-data', 'litellm',
  '--collect-data', 'akshare',
  '--hidden-import=tiktoken_ext.openai_public',
  '--exclude', 'PyQt5',
  '--exclude', 'PySide6',
  '--exclude', 'PyQt6',
  '--exclude', 'qtpy'
)
$pyInstallerArgs += $hiddenImportArgs
$pyInstallerArgs += 'main.py'

Write-Host "Running: $pythonBin $($pyInstallerArgs -join ' ')"
& $pythonBin @pyInstallerArgs
if ($LASTEXITCODE -ne 0) {
  throw "PyInstaller failed with exit code $LASTEXITCODE."
}

if (!(Test-Path 'dist\stock_analysis')) {
  throw 'PyInstaller finished but dist\stock_analysis was not generated.'
}

# 冒烟测试：验证打包后的 exe 能否正确导入关键模块
Write-Host 'Running smoke test on packaged exe...'
$exePath = Join-Path (Get-Location) 'dist\stock_analysis\stock_analysis.exe'
$smokeTestCode = @'
import tiktoken
try:
    enc = tiktoken.get_encoding("cl100k_base")
    print("SMOKE_OK: tiktoken encoding loaded")
except Exception as e:
    print("SMOKE_FAIL: tiktoken - " + str(e))
    raise SystemExit(1)

# 验证 api.app 能正常导入（FastAPI 应用初始化）
try:
    import api.app
    print("SMOKE_OK: api.app imported")
except Exception as e:
    print("SMOKE_FAIL: api.app - " + str(e))
    raise SystemExit(1)
'@
$smokeResult = & $exePath -c $smokeTestCode 2>&1
if ($LASTEXITCODE -ne 0) {
  Write-Host "Smoke test FAILED:"
  Write-Host $smokeResult
  throw 'Smoke test failed - packaged exe has missing dependencies. Check the error above and add --hidden-import for the missing module.'
}
Write-Host "Smoke test PASSED: $smokeResult"

Copy-Item -Path 'dist\stock_analysis' -Destination 'dist\backend\stock_analysis' -Recurse -Force

Write-Host 'Backend build completed.'
