<#
用途：粗粒度二分类SFT统一CoT生成入口；代码和Prompt相同，只切换输入/输出目录。
Target：baseline=原1,500条；pool=全部未采样候选；step5=长度去偏实验100条。
是否调用API：正式运行会；-DryRun不会。无需GPU，三个Target使用各自断点。
当前命令：
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\data_pipeline\coarse_sft\run_cot.ps1 -Target step5 -DryRun
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\data_pipeline\coarse_sft\run_cot.ps1 -Target step5
#>

param(
    [ValidateSet("baseline", "pool", "step5")][string]$Target = "step5",
    [switch]$DryRun,
    [switch]$Review,
    [int]$Limit = 0,
    [ValidateSet("all", "train", "validation", "test")][string]$Split = "all"
)

$ErrorActionPreference = "Stop"

# 三个Target共用同一套模型和限流配置。
$Model = "gpt-6.1-sol"
$BaseUrl = "https://new.mrxiwang.eu.cc/v1"
$MaxTokens = "8192"
$Workers = "4"
$ReviewModels = "gpt-6-sol,gpt-6-astra"
$RequestInterval = "8"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = (Get-Command python.exe -ErrorAction Stop).Source
}

# 优先使用环境变量；本机兼容读取已被Git忽略的Step 3配置，不执行该脚本。
if (-not $env:STEP4_API_KEY) {
    $env:STEP4_API_KEY = $env:STEP3_API_KEY
    $KeyFile = Join-Path $PSScriptRoot "run_step3_company_full.local.ps1"
    if (-not $env:STEP4_API_KEY -and (Test-Path -LiteralPath $KeyFile)) {
        $Match = [regex]::Match((Get-Content -LiteralPath $KeyFile -Raw), '(?m)^\$env:STEP3_API_KEY\s*=\s*"([^"]+)"')
        if ($Match.Success) { $env:STEP4_API_KEY = $Match.Groups[1].Value }
    }
}
$env:STEP4_MODEL = $Model
$env:STEP4_BASE_URL = $BaseUrl
$env:STEP4_MAX_TOKENS = $MaxTokens
$env:STEP4_WORKERS = $Workers
$env:STEP4_REVIEW_MODELS = $ReviewModels
$env:STEP4_REQUEST_INTERVAL = $RequestInterval
$env:STEP4_REASONING_EFFORT = "high"
$env:STEP4_STREAM = "true"
$env:STEP4_MAX_TOKENS_FIELD = "max_completion_tokens"
$env:STEP4_RETRIES = "1"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"

if ($Limit -lt 0) { throw "Limit must be >= 0" }
if ($Review -and $Target -ne "baseline") { throw "-Review仅用于Target=baseline。" }

Set-Location $ProjectRoot
switch ($Target) {
    "baseline" { $PipelineArgs = @("data_pipeline/coarse_sft/pipeline.py", "step4") }
    "pool" { $PipelineArgs = @("data_pipeline/coarse_sft/pipeline.py", "cot_pool", "--split", $Split) }
    "step5" { $PipelineArgs = @("data_pipeline/coarse_sft/pipeline.py", "cot_pool", "--selection", "step5") }
}
if ($DryRun) { $PipelineArgs += "--dry-run" }
if ($Review) { $PipelineArgs += "--review" }
if ($Limit -gt 0) { $PipelineArgs += @("--limit", "$Limit") }

Write-Host "CoT generation: target=$Target model=$Model workers=$Workers interval=$RequestInterval"
& $Python @PipelineArgs
if ($LASTEXITCODE -ne 0) { throw "CoT generation stopped. Completed responses remain in the target checkpoint." }

if ($Target -eq "step5" -and -not $DryRun) {
    & $Python "data_pipeline/coarse_sft/pipeline.py" "step5"
    if ($LASTEXITCODE -ne 0) { throw "CoT generated, but Step 5 selection report refresh failed." }
}
