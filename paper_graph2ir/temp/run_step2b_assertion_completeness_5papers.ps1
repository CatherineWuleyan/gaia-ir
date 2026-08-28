# run_step2b_assertion_completeness_5papers.ps1
#
# 放在 paper_graph2ir 项目根目录下(跟 claude_api_call.py 同级)。
# 对这5篇论文依次跑 step2_claim_completeness\step2b_check_assertion_completeness.py,
# 对 step2a 判定为"不是纯实验数据"的每条claim,逐条问"忽略语法/指代问题,
# 这是不是一个完整的断言",把结果写回 claim_completeness_analysis.json。
#
# 前置条件:这5篇论文的 data\<paper_id>\claim_completeness_analysis.json
# 必须已经存在(即已经跑完 step2a_check_pure_data.py)——step2b本身会检查
# 这一点,找不到会直接报错退出,这里不重复做前置校验。
#
# 注意:step2b是"原地更新"这个文件(在已有记录上新增 是完整断言/
# completeness_status 两个字段),不是像step2a那样另外产生一份新文件,
# 所以这次备份格外重要——万一中途出问题,备份是唯一能拿回step2a阶段
# 结果的办法。
#
# 跟step2a一样是同步、逐条调用,没有Batch API版本。这一步是一条一条单独
# 问的(不是按conclusion批量问),所以调用次数等于"不是纯实验数据"的claim
# 总数,不是conclusion数量——数量比step2a那一轮多不少,跑的时间也会更长。
#
# 默认处理全部5篇:
#   1032903864883347458 / 867750889056633390 / 867752822639165809 /
#   867760083600146646 / 867757662605934651
#
# 用法:
#   .\run_step2b_assertion_completeness_5papers.ps1
#   .\run_step2b_assertion_completeness_5papers.ps1 -PaperIds "id1","id2"
#   .\run_step2b_assertion_completeness_5papers.ps1 -NoBackup

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
$Step2bScript = Join-Path $ScriptDir "step2_claim_completeness\step2b_check_assertion_completeness.py"

if (-not (Test-Path $Step2bScript)) {
    Write-Host "找不到文件: $Step2bScript" -ForegroundColor Red
    exit 1
}

foreach ($paperId in $PaperIds) {
    $f = Join-Path $ScriptDir "data\$paperId\claim_completeness_analysis.json"
    if (-not (Test-Path $f)) {
        Write-Host "找不到文件: $f (paper_id=$paperId 需要先跑完 step2a_check_pure_data.py)" -ForegroundColor Red
        exit 1
    }
}

# --- 备份旧的 claim_completeness_analysis.json(这一步是原地更新,备份格外重要) ---
if (-not $NoBackup) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    foreach ($paperId in $PaperIds) {
        $f = Join-Path $ScriptDir "data\$paperId\claim_completeness_analysis.json"
        $bak = "$f.bak_$stamp"
        Copy-Item -Path $f -Destination $bak
        Write-Host "已备份旧文件: $bak" -ForegroundColor DarkGray
    }
}

Write-Host ""
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "step2b_check_assertion_completeness.py (逐篇同步处理 $($PaperIds.Count) 篇论文)" -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "注意: 这一步是逐条claim单独调用,数量比step2a那一轮多,可能会跑一阵子。" -ForegroundColor DarkYellow

foreach ($paperId in $PaperIds) {
    Write-Host ""
    Write-Host "--- $paperId ---" -ForegroundColor Cyan
    python $Step2bScript $paperId
    if ($LASTEXITCODE -ne 0) {
        Write-Host "step2b_check_assertion_completeness.py 失败(paper_id=$paperId),退出码 $LASTEXITCODE" -ForegroundColor Red
        exit 1
    }
}

# --- 跑完之后打印结果摘要 ---
Write-Host ""
Write-Host "===================================================================" -ForegroundColor Green
Write-Host "全部完成,结果摘要:" -ForegroundColor Green
Write-Host "===================================================================" -ForegroundColor Green

$grandTotal = 0
$grandPureData = 0
$grandComplete = 0
$grandIncomplete = 0
$grandFallback = 0

foreach ($paperId in $PaperIds) {
    $f = Join-Path $ScriptDir "data\$paperId\claim_completeness_analysis.json"
    Write-Host ""
    Write-Host "$paperId" -ForegroundColor Cyan

    $results = @(Get-Content -Path $f -Raw -Encoding UTF8 | ConvertFrom-Json)

    $pureDataCount = @($results | Where-Object { $_.'是纯实验数据' -eq $true }).Count
    $completeCount = @($results | Where-Object { $_.'是完整断言' -eq $true }).Count
    $incompleteCount = @($results | Where-Object { $_.'是完整断言' -eq $false }).Count
    $fallbackCount = @($results | Where-Object { $_.completeness_status -eq "fallback_trivial" }).Count

    Write-Host "  共 $($results.Count) 条claim"
    Write-Host "    纯实验数据(跳过本步): $pureDataCount 条"
    Write-Host "    完整断言: $completeCount 条"
    Write-Host "    不完整断言(留给下一步): $incompleteCount 条"
    Write-Host "    其中走了保守兜底(fallback_trivial): $fallbackCount 条"

    $grandTotal += $results.Count
    $grandPureData += $pureDataCount
    $grandComplete += $completeCount
    $grandIncomplete += $incompleteCount
    $grandFallback += $fallbackCount
}

Write-Host ""
Write-Host "5篇论文合计: $grandTotal 条claim" -ForegroundColor Cyan
Write-Host "  纯实验数据(跳过): $grandPureData 条" -ForegroundColor Cyan
Write-Host "  完整断言: $grandComplete 条" -ForegroundColor Cyan
Write-Host "  不完整断言(留给下一步分析): $grandIncomplete 条" -ForegroundColor Cyan
Write-Host "  保守兜底(fallback_trivial): $grandFallback 条" -ForegroundColor Cyan
