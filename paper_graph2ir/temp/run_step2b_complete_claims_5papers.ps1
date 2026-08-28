# run_step2b_complete_claims_5papers.ps1
#
# 放在 paper_graph2ir 项目根目录下(跟 claude_api_call.py 同级)。
# 对这5篇论文依次跑 step2_claim_completeness\step2b_complete_claims.py,
# 对 step2a 判定为"不是纯实验数据"的每条claim,产出"不依赖任何未引用内容"
# 的完整表述(取词/引用限定在assertion/论据/example/elaboration/framing/
# other/motivation这几类;论证/instance/relation/connection不允许),
# 把结果写回 claim_completeness_analysis.json。
#
# 前置条件: data\<paper_id>\claim_completeness_analysis.json 必须已经
# 存在(即已经跑完 step2a_check_pure_data.py)。
#
# 注意:step2b_complete_claims.py是"原地更新"这个文件(在已有记录上新增
# 完整表述/需要更多上下文/completion_status三个字段),不是像step2a那样
# 另外产生一份新文件,所以这次备份格外重要——万一中途出问题,备份是
# 唯一能拿回step2a阶段结果的办法。
#
# 这一步比step2a贵得多:一个conclusion一次调用,用Sonnet 5+思考(effort=
# high),失败时最多再升到Opus 5、再升到Sonnet 5(不开思考,更大token
# 预算)——实测下来质量很扎实,但相应地也会跑得比step2a慢、调用成本更高。
#
# 默认处理全部5篇:
#   1032903864883347458 / 867750889056633390 / 867752822639165809 /
#   867760083600146646 / 867757662605934651
#
# 用法:
#   .\run_step2b_complete_claims_5papers.ps1
#   .\run_step2b_complete_claims_5papers.ps1 -PaperIds "id1","id2"
#   .\run_step2b_complete_claims_5papers.ps1 -NoBackup

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
$Step2bScript = Join-Path $ScriptDir "step2_claim_completeness\step2b_complete_claims.py"

if (-not (Test-Path $Step2bScript)) {
    Write-Host "找不到文件: $Step2bScript" -ForegroundColor Red
    exit 1
}

foreach ($paperId in $PaperIds) {
    $organizedPath = Join-Path $ScriptDir "data\$paperId\organized_content.json"
    $completenessPath = Join-Path $ScriptDir "data\$paperId\claim_completeness_analysis.json"
    if (-not (Test-Path $organizedPath)) {
        Write-Host "找不到文件: $organizedPath (paper_id=$paperId 需要先跑完 step1b_organize_labeled_content.py)" -ForegroundColor Red
        exit 1
    }
    if (-not (Test-Path $completenessPath)) {
        Write-Host "找不到文件: $completenessPath (paper_id=$paperId 需要先跑完 step2a_check_pure_data.py)" -ForegroundColor Red
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
Write-Host "step2b_complete_claims.py (逐篇同步处理 $($PaperIds.Count) 篇论文)" -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "注意: 这一步用Sonnet 5+思考(effort=high),比step2a贵、也慢很多,可能要跑一阵子。" -ForegroundColor DarkYellow

foreach ($paperId in $PaperIds) {
    Write-Host ""
    Write-Host "--- $paperId ---" -ForegroundColor Cyan
    python $Step2bScript $paperId
    if ($LASTEXITCODE -ne 0) {
        Write-Host "step2b_complete_claims.py 失败(paper_id=$paperId),退出码 $LASTEXITCODE" -ForegroundColor Red
        exit 1
    }
}

# --- 跑完之后打印结果摘要 ---
Write-Host ""
Write-Host "===================================================================" -ForegroundColor Green
Write-Host "全部完成,结果摘要:" -ForegroundColor Green
Write-Host "===================================================================" -ForegroundColor Green

$grandTotal = 0
$grandCompleted = 0
$grandNeedsContext = 0
$grandFallback = 0

foreach ($paperId in $PaperIds) {
    $f = Join-Path $ScriptDir "data\$paperId\claim_completeness_analysis.json"
    Write-Host ""
    Write-Host "$paperId" -ForegroundColor Cyan

    $results = @(Get-Content -Path $f -Raw -Encoding UTF8 | ConvertFrom-Json)

    $completedResults = @($results | Where-Object { $null -ne $_.'完整表述' })
    $completedCount = $completedResults.Count
    $needsContextCount = @($completedResults | Where-Object { $_.'需要更多上下文' -and $_.'需要更多上下文'.Count -gt 0 }).Count

    $byStatus = @($completedResults | Where-Object { $_.completion_status }) | Group-Object completion_status
    Write-Host "  共 $($results.Count) 条claim,其中 $completedCount 条完成了补全"
    foreach ($g in $byStatus) {
        Write-Host "    状态 $($g.Name): $($g.Count) 条"
    }
    Write-Host "    带有'需要更多上下文'标记: $needsContextCount 条"

    $fallbackCount = @($completedResults | Where-Object { $_.completion_status -eq "fallback_original" }).Count

    $grandTotal += $results.Count
    $grandCompleted += $completedCount
    $grandNeedsContext += $needsContextCount
    $grandFallback += $fallbackCount
}

Write-Host ""
Write-Host "5篇论文合计: $grandTotal 条claim,其中 $grandCompleted 条完成了补全" -ForegroundColor Cyan
Write-Host "  带有'需要更多上下文'标记: $grandNeedsContext 条" -ForegroundColor Cyan
Write-Host "  保守兜底(fallback_original): $grandFallback 条" -ForegroundColor Cyan
