<#
用途：细粒度三标签SFT的Step 2，为4,000条canonical轨迹生成CoT并最终导出Parquet。
当前是否需要：需要；细粒度CoT尚未全量完成时运行本脚本。
是否调用API：正式运行会；-DryRun不会。无需GPU，支持断点续跑。
输入：data/1.5/fine_sft/step1/final/*.jsonl
输出：data/1.5/fine_sft/step2/下的断点、报告和最终数据。
常用命令：
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\data_pipeline\fine_sft\run_step2.ps1 -DryRun
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\data_pipeline\fine_sft\run_step2.ps1 -Limit 6
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\data_pipeline\fine_sft\run_step2.ps1
#>

param(
    [switch]$DryRun,
    [int]$Limit = 0
)

$ErrorActionPreference = "Stop"

# No GPU is required. Edit model, concurrency, and rate limit here.
$Model = "gpt-6.1-sol"
$BaseUrl = "https://new.mrxiwang.eu.cc/v1"
$MaxTokens = "8192"
$Workers = "4"
$RequestInterval = "8"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = (Get-Command python.exe -ErrorAction Stop).Source
}

# Reuse a local coarse-stage key when available. This script contains no secret.
if (-not $env:FINE_STEP2_API_KEY) { $env:FINE_STEP2_API_KEY = $env:STEP4_API_KEY }
if (-not $env:FINE_STEP2_API_KEY) { $env:FINE_STEP2_API_KEY = $env:STEP3_API_KEY }
if (-not $env:FINE_STEP2_API_KEY) {
    $KeyFile = Join-Path $ProjectRoot "data_pipeline\coarse_sft\run_step3_company_full.local.ps1"
    if (Test-Path -LiteralPath $KeyFile) {
        $Match = [regex]::Match((Get-Content -LiteralPath $KeyFile -Raw), '(?m)^\$env:STEP3_API_KEY\s*=\s*"([^"]+)"')
        if ($Match.Success) { $env:FINE_STEP2_API_KEY = $Match.Groups[1].Value }
    }
}
if (-not $DryRun -and -not $env:FINE_STEP2_API_KEY) {
    throw 'Set $env:FINE_STEP2_API_KEY before running Step 2.'
}

$env:FINE_STEP2_MODEL = $Model
$env:FINE_STEP2_BASE_URL = $BaseUrl
$env:FINE_STEP2_MAX_TOKENS = $MaxTokens
$env:FINE_STEP2_WORKERS = $Workers
$env:FINE_STEP2_REQUEST_INTERVAL = $RequestInterval
$env:FINE_STEP2_REASONING_EFFORT = "high"
$env:FINE_STEP2_STREAM = "true"
$env:FINE_STEP2_RETRIES = "1"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"

Set-Location $ProjectRoot
$PipelineArgs = @("data_pipeline/fine_sft/pipeline.py", "step2")
if ($DryRun) { $PipelineArgs += "--dry-run" }
if ($Limit -gt 0) { $PipelineArgs += @("--limit", "$Limit") }
Write-Host "Fine Step 2: model=$Model max_tokens=$MaxTokens workers=$Workers global_interval=$RequestInterval"
& $Python @PipelineArgs
if ($LASTEXITCODE -ne 0) { throw "Fine Step 2 stopped. See the error above." }
