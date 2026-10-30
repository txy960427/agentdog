<#
用途：粗粒度二分类SFT统一CoT语义审核入口；审核Prompt和逻辑相同，只切换数据目录。
Target：baseline=审核原1,500条；step5=审核长度去偏实验最终100条。
是否调用API：正式运行会；-DryRun不会。无需GPU，审核结果写入各Target自己的check/目录。
当前命令：
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\data_pipeline\coarse_sft\run_cot_check.ps1 -Target step5 -DryRun
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\data_pipeline\coarse_sft\run_cot_check.ps1 -Target step5
#>

param(
    [ValidateSet("baseline", "step5")][string]$Target = "step5",
    [switch]$DryRun,
    [int]$Limit = 0
)

$ErrorActionPreference = "Stop"

$Model = "gpt-6-astra"
$BaseUrl = "https://new.mrxiwang.eu.cc/v1"
$MaxTokens = "8192"
$Workers = "4"
$RequestInterval = "8"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = (Get-Command python.exe -ErrorAction Stop).Source
}

if (-not $env:STEP4_API_KEY) {
    $env:STEP4_API_KEY = $env:STEP3_API_KEY
    $KeyFile = Join-Path $PSScriptRoot "run_step3_company_full.local.ps1"
    if (-not $env:STEP4_API_KEY -and (Test-Path -LiteralPath $KeyFile)) {
        $Match = [regex]::Match((Get-Content -LiteralPath $KeyFile -Raw), '(?m)^\$env:STEP3_API_KEY\s*=\s*"([^"]+)"')
        if ($Match.Success) { $env:STEP4_API_KEY = $Match.Groups[1].Value }
    }
}
$env:STEP4_CHECK_MODEL = $Model
$env:STEP4_BASE_URL = $BaseUrl
$env:STEP4_MAX_TOKENS = $MaxTokens
$env:STEP4_WORKERS = $Workers
$env:STEP4_REQUEST_INTERVAL = $RequestInterval
$env:STEP4_REASONING_EFFORT = "high"
$env:STEP4_STREAM = "true"
$env:STEP4_MAX_TOKENS_FIELD = "max_completion_tokens"
$env:STEP4_RETRIES = "1"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"

if ($Limit -lt 0) { throw "Limit must be >= 0" }
Set-Location $ProjectRoot
$PipelineArgs = @("data_pipeline/coarse_sft/pipeline.py", "step4", "--check", "--target", $Target)
if ($DryRun) { $PipelineArgs += "--dry-run" }
if ($Limit -gt 0) { $PipelineArgs += @("--limit", "$Limit") }

Write-Host "CoT check: target=$Target model=$Model workers=$Workers interval=$RequestInterval"
& $Python @PipelineArgs
if ($LASTEXITCODE -ne 0) { throw "CoT check stopped. Completed results remain in the target check directory." }
