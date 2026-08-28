# run_step2a_pure_data_5papers.ps1
#
# 放在 paper_graph2ir 项目根目录下(跟 claude_api_call.py 同级)。
# 对这5篇论文依次跑 step2_claim_completeness\step2a_check_pure_data.py,
# 判断每条claim(organized_content.json里label为assertion/论据/example的
# part)是不是"纯实验数据"。
#
# 前置条件:这5篇论文的 data\<paper_id>\organized_content.json 必须已经
# 存在(即已经跑完 step1b_organize_labeled_content.py)——step2a_check_pure_data.py
# 本身会检查这一点,找不到会直接报错退出,这里不重复做前置校验。
#
# 跟 run_batch_pipeline_4papers.ps1 不同:step2a目前只有单条同步调用版本,
# 没有像step1a那样另外做一份Batch API版本,所以这里是逐篇顺序调用,没有
# "提交后台任务、等结果"这一说;跑多久取决于每篇论文里含claim的conclusion
# 数量(每个conclusion一次API调用,失败时最多再多打一次Sonnet 5)。
#
# 默认处理全部5篇:
#   1032903864883347458 / 867750889056633390 / 867752822639165809 /
#   867760083600146646 / 867757662605934651
#
# 用法:
#   .\run_step2a_pure_data_5papers.ps1
#   .\run_step2a_pure_data_5papers.ps1 -PaperIds "id1","id2"   # 处理别的论文列表
#   .\run_step2a_pure_data_5papers.ps1 -NoBackup               # 不备份旧数据,直接覆盖

param(
    [string[]]$PaperIds = @(
        "1032903864883347458",
        "867750889056633390",
        "867752822639165809",
        "867760083600146646",
        "867757662605934651"
    ),
    [switch]$NoBackup
)

$ErrorActionPreference = "Stop"

if (-not $env:ANTHROPIC_API_KEY) {
    Write-Host "没检测到 ANTHROPIC_API_KEY 环境变量,先设置好再跑。" -ForegroundColor Red
    exit 1
}

$ScriptDir = $PSScriptRoot
$Step2aScript = Join-Path $ScriptDir "step2_claim_completeness\step2a_check_pure_data.py"

if (-not (Test-Path $Step2aScript)) {
    Write-Host "找不到文件: $Step2aScript" -ForegroundColor Red
    exit 1
}

foreach ($paperId in $PaperIds) {
    $dataDir = Join-Path $ScriptDir "data\$paperId"
    $organizedPath = Join-Path $dataDir "organized_content.json"
    if (-not (Test-Path $organizedPath)) {
        Write-Host "找不到文件: $organizedPath (paper_id=$paperId 需要先跑完 step1b_organize_labeled_content.py)" -ForegroundColor Red
        exit 1
    }
}

# --- 备份旧的 claim_completeness_analysis.json(如果存在) ---
if (-not $NoBackup) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    foreach ($paperId in $PaperIds) {
        $f = Join-Path $ScriptDir "data\$paperId\claim_completeness_analysis.json"
        if (Test-Path $f) {
            $bak = "$f.bak_$stamp"
            Copy-Item -Path $f -Destination $bak
            Write-Host "已备份旧文件: $bak" -ForegroundColor DarkGray
        }
    }
}

Write-Host ""
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "step2a_check_pure_data.py (逐篇同步处理 $($PaperIds.Count) 篇论文)" -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan

foreach ($paperId in $PaperIds) {
    Write-Host ""
    Write-Host "--- $paperId ---" -ForegroundColor Cyan
    python $Step2aScript $paperId
    if ($LASTEXITCODE -ne 0) {
        Write-Host "step2a_check_pure_data.py 失败(paper_id=$paperId),退出码 $LASTEXITCODE" -ForegroundColor Red
        exit 1
    }
}

# --- 跑完之后打印结果摘要 ---
Write-Host ""
Write-Host "===================================================================" -ForegroundColor Green
Write-Host "全部完成,结果摘要:" -ForegroundColor Green
Write-Host "===================================================================" -ForegroundColor Green

$grandTotal = 0
$grandPure = 0
$grandFallback = 0

foreach ($paperId in $PaperIds) {
    $f = Join-Path $ScriptDir "data\$paperId\claim_completeness_analysis.json"
    Write-Host ""
    Write-Host "$paperId" -ForegroundColor Cyan

    $results = Get-Content -Path $f -Raw -Encoding UTF8 | ConvertFrom-Json

    $byStatus = $results | Group-Object pure_data_status
    foreach ($g in $byStatus) {
        Write-Host "  状态 $($g.Name): $($g.Count) 条"
    }

    $pureCount = ($results | Where-Object { $_.'是纯实验数据' -eq $true }).Count
    Write-Host "  共 $($results.Count) 条claim,其中判定为纯实验数据 $pureCount 条"

    $grandTotal += $results.Count
    $grandPure += $pureCount
    $grandFallback += ($results | Where-Object { $_.pure_data_status -eq "fallback_trivial" }).Count
}

Write-Host ""
Write-Host "5篇论文合计: $grandTotal 条claim,其中 $grandPure 条为纯实验数据,$grandFallback 条走了保守兜底(fallback_trivial)" -ForegroundColor Cyan
