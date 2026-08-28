# run_step2d_remove_instance_content_remaining4papers.ps1
#
# 放在 paper_graph2ir 项目根目录下(跟 claude_api_call.py、
# run_step2d_remove_instance_content.ps1 同级)。
#
# 867752822639165809 已经手工跑过、人工核对过质量了,这个脚本跑剩下的
# 4篇论文(前提是这4篇都已经跑过 step2c_check_instance_containment.py,
# claim_completeness_analysis.json里得有 instance_containment 字段):
#   1032903864883347458
#   867750889056633390
#   867757662605934651
#   867760083600146646
#
# 每篇论文调一次 step2d_remove_instance_content.py(真实调用API),单篇
# 失败不会中断其余论文(打印错误、跳到下一篇),最后打印5篇论文(含已经
# 跑过的867752822639165809)的汇总统计。
#
# 用法:
#   .\run_step2d_remove_instance_content_remaining4papers.ps1

$ErrorActionPreference = "Stop"

if (-not $env:ANTHROPIC_API_KEY) {
    Write-Host "没检测到 ANTHROPIC_API_KEY 环境变量,先设置好再跑。" -ForegroundColor Red
    exit 1
}

$ScriptDir = $PSScriptRoot
$Step2dScript = Join-Path $ScriptDir "step2_claim_completeness\step2d_remove_instance_content.py"

if (-not (Test-Path $Step2dScript)) {
    Write-Host "找不到文件: $Step2dScript" -ForegroundColor Red
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

    $existing = Get-Content -Path $ClaimAnalysisPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $hasContainmentField = $existing | Where-Object { $_.PSObject.Properties.Name -contains "instance_containment" } | Select-Object -First 1
    if (-not $hasContainmentField) {
        Write-Host "跳过 ${id}: $ClaimAnalysisPath 里还没有 instance_containment 字段(需要先跑 step2c)" -ForegroundColor Yellow
        $Failed += $id
        continue
    }

    Write-Host ""
    Write-Host "===================================================================" -ForegroundColor Cyan
    Write-Host "step2d_remove_instance_content.py (paper_id=$id)" -ForegroundColor Cyan
    Write-Host "===================================================================" -ForegroundColor Cyan

    python $Step2dScript $id

    if ($LASTEXITCODE -ne 0) {
        Write-Host "${id} 失败,退出码 $LASTEXITCODE,继续跑下一篇" -ForegroundColor Red
        $Failed += $id
    }
}

# --- 5篇论文(含之前已经跑过的867752822639165809)的汇总统计 ---
Write-Host ""
Write-Host "===================================================================" -ForegroundColor Green
Write-Host "全部完成,5篇论文汇总:" -ForegroundColor Green
Write-Host "===================================================================" -ForegroundColor Green

$AllPaperIds = @("867752822639165809") + $RemainingPaperIds
$TotalConfirmed = 0
$TotalRejected = 0
$TotalFallback = 0

foreach ($id in $AllPaperIds) {
    $ClaimAnalysisPath = Join-Path $ScriptDir "data\$id\claim_completeness_analysis.json"
    if (-not (Test-Path $ClaimAnalysisPath)) {
        continue
    }
    $records = Get-Content -Path $ClaimAnalysisPath -Raw -Encoding UTF8 | ConvertFrom-Json

    $confirmed = @($records | Where-Object { $_.instance_containment_confirmed -eq $true }).Count
    $rejected = @($records | Where-Object { $_.instance_containment -eq $true -and $_.instance_containment_confirmed -eq $false }).Count
    $fallback = @($records | Where-Object { $_.instance_removal_status -eq "fallback_unchanged" }).Count

    Write-Host "  ${id}: $confirmed 条确认混入并剥离,$rejected 条复核后判定不是混入,$fallback 条走了保守兜底"

    $TotalConfirmed += $confirmed
    $TotalRejected += $rejected
    $TotalFallback += $fallback
}

Write-Host ""
Write-Host "合计: $TotalConfirmed 条确认混入并剥离,$TotalRejected 条复核后判定不是混入,$TotalFallback 条走了保守兜底"

if ($Failed.Count -gt 0) {
    Write-Host ""
    Write-Host "以下论文没跑成功: $($Failed -join ', ')" -ForegroundColor Red
}
