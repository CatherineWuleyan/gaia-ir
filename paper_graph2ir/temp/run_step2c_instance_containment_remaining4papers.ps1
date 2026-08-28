# run_step2c_instance_containment_remaining4papers.ps1
#
# 放在 paper_graph2ir 项目根目录下(跟 claude_api_call.py、
# run_step2c_instance_containment.ps1 同级)。
#
# 867752822639165809 已经手工跑过、人工核对过质量了,这个脚本跑剩下的
# 4篇论文:
#   1032903864883347458
#   867750889056633390
#   867757662605934651
#   867760083600146646
#
# 每篇论文调一次 step2c_check_instance_containment.py(真实调用API),
# 单篇失败不会中断其余论文(打印错误、跳到下一篇),最后打印5篇论文
# (含已经跑过的867752822639165809)的汇总统计。
#
# 用法:
#   .\run_step2c_instance_containment_remaining4papers.ps1

$ErrorActionPreference = "Stop"

if (-not $env:ANTHROPIC_API_KEY) {
    Write-Host "没检测到 ANTHROPIC_API_KEY 环境变量,先设置好再跑。" -ForegroundColor Red
    exit 1
}

$ScriptDir = $PSScriptRoot
$Step2cScript = Join-Path $ScriptDir "step2_claim_completeness\step2c_check_instance_containment.py"

if (-not (Test-Path $Step2cScript)) {
    Write-Host "找不到文件: $Step2cScript" -ForegroundColor Red
    exit 1
}

$RemainingPaperIds = @(
    "1032903864883347458",
    "867750889056633390",
    "867757662605934651",
    "867760083600146646"
)

$Failed = @()

foreach ($id in $RemainingPaperIds) {
    $OrganizedPath = Join-Path $ScriptDir "data\$id\organized_content.json"
    $ClaimAnalysisPath = Join-Path $ScriptDir "data\$id\claim_completeness_analysis.json"

    if (-not (Test-Path $OrganizedPath)) {
        Write-Host "跳过 ${id}: 找不到 $OrganizedPath" -ForegroundColor Yellow
        $Failed += $id
        continue
    }
    if (-not (Test-Path $ClaimAnalysisPath)) {
        Write-Host "跳过 ${id}: 找不到 $ClaimAnalysisPath" -ForegroundColor Yellow
        $Failed += $id
        continue
    }

    Write-Host ""
    Write-Host "===================================================================" -ForegroundColor Cyan
    Write-Host "step2c_check_instance_containment.py (paper_id=$id)" -ForegroundColor Cyan
    Write-Host "===================================================================" -ForegroundColor Cyan

    python $Step2cScript $id

    if ($LASTEXITCODE -ne 0) {
        Write-Host "$id 失败,退出码 $LASTEXITCODE,继续跑下一篇" -ForegroundColor Red
        $Failed += $id
    }
}

# --- 5篇论文(含之前已经跑过的867752822639165809)的汇总统计 ---
Write-Host ""
Write-Host "===================================================================" -ForegroundColor Green
Write-Host "全部完成,5篇论文汇总:" -ForegroundColor Green
Write-Host "===================================================================" -ForegroundColor Green

$AllPaperIds = @("867752822639165809") + $RemainingPaperIds
$TotalTested = 0
$TotalTrue = 0
$TotalFallback = 0

foreach ($id in $AllPaperIds) {
    $ClaimAnalysisPath = Join-Path $ScriptDir "data\$id\claim_completeness_analysis.json"
    if (-not (Test-Path $ClaimAnalysisPath)) {
        continue
    }
    $records = Get-Content -Path $ClaimAnalysisPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $tested = $records | Where-Object { $_.instance_containment_status -notin @("skipped_no_instances", "skipped_pure_data") }
    $trueCount = $tested | Where-Object { $_.instance_containment -eq $true }
    $fallbackCount = $tested | Where-Object { $_.instance_containment_status -eq "fallback_trivial" }

    $n_tested = @($tested).Count
    $n_true = @($trueCount).Count
    $n_fallback = @($fallbackCount).Count

    Write-Host "  $id : $n_tested 条测试,$n_true 条True,$n_fallback 条走了保守兜底"

    $TotalTested += $n_tested
    $TotalTrue += $n_true
    $TotalFallback += $n_fallback
}

Write-Host ""
Write-Host "合计: $TotalTested 条测试,$TotalTrue 条True,$TotalFallback 条走了保守兜底"

if ($Failed.Count -gt 0) {
    Write-Host ""
    Write-Host "以下论文没跑成功: $($Failed -join ', ')" -ForegroundColor Red
}
