# run_debug_step2b_one_conclusion.ps1
#
# 放在 paper_graph2ir 项目根目录下(跟 claude_api_call.py 同级)。
# 跑 step2_claim_completeness\debug_step2b_complete_claims.py,对一个指定
# 的conclusion走完整的"补全完整表述"三档流程(Sonnet5思考medium ->
# Opus5不思考 -> Sonnet5不思考,失败则原文照抄兜底),只读、不写文件,
# 方便反复试同一个conclusion而不用担心覆盖已有结果。
#
# 默认跑 1032903864883347458 论文的 conclusion_2——这个conclusion比较
# 有代表性: [6]带密集LaTeX公式,[10]+[11]是经典的"claim[11]需要引用
# claim[10]整句话才能补全"的案例,还混着other/framing/connection/
# elaboration好几种非claim类型的上下文。
#
# 前置条件: data\<paper_id>\claim_completeness_analysis.json 必须已经
# 存在(即已经跑完 step2a_check_pure_data.py)。
#
# 用法:
#   .\run_debug_step2b_one_conclusion.ps1
#   .\run_debug_step2b_one_conclusion.ps1 -PaperId "id" -Conclusion "conclusion_3"
#   .\run_debug_step2b_one_conclusion.ps1 -Conclusion 0   # 按索引选,而不是短id

param(
    [string]$PaperId = "1032903864883347458",
    [string]$Conclusion = "conclusion_2"
)

$ErrorActionPreference = "Stop"

if (-not $env:ANTHROPIC_API_KEY) {
    Write-Host "没检测到 ANTHROPIC_API_KEY 环境变量,先设置好再跑。" -ForegroundColor Red
    exit 1
}

$ScriptDir = $PSScriptRoot
$DebugScript = Join-Path $ScriptDir "step2_claim_completeness\debug_step2b_complete_claims.py"

if (-not (Test-Path $DebugScript)) {
    Write-Host "找不到文件: $DebugScript" -ForegroundColor Red
    exit 1
}

$completenessPath = Join-Path $ScriptDir "data\$PaperId\claim_completeness_analysis.json"
if (-not (Test-Path $completenessPath)) {
    Write-Host "找不到文件: $completenessPath (需要先跑完 step2a_check_pure_data.py)" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "debug_step2b_complete_claims.py (只读,不写文件)" -ForegroundColor Cyan
Write-Host "论文: $PaperId   conclusion: $Conclusion" -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host ""

python $DebugScript $PaperId $Conclusion
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "运行失败,退出码 $LASTEXITCODE" -ForegroundColor Red
    exit 1
}
