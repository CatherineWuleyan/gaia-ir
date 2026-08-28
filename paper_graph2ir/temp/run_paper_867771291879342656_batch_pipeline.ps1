# run_paper_867771291879342656_batch_pipeline.ps1
#
# 放在 paper_graph2ir 项目根目录下(跟 claude_api_call.py、
# run_multi_paper_step1a_batch.py 同级)。
#
# 对 867771291879342656 这一篇论文跑完整14步流水线,step1a走Batch API
# (run_multi_paper_step1a_batch.py),后面13步(step1b~step4d)不变。
#
# 前提: data\867771291879342656\graph.json 必须已经存在(这是最上游的
# 输入,不是流水线任何一步自己产出的)。
#
# 用法:
#   .\run_paper_867771291879342656_batch_pipeline.ps1

$ErrorActionPreference = "Stop"

$PaperId = "867771291879342656"

if (-not $env:ANTHROPIC_API_KEY) {
    Write-Host "没检测到 ANTHROPIC_API_KEY 环境变量,先设置好再跑。" -ForegroundColor Red
    exit 1
}

$ScriptDir = $PSScriptRoot
$BatchScript = Join-Path $ScriptDir "run_multi_paper_step1a_batch.py"
$GraphPath = Join-Path $ScriptDir "data\$PaperId\graph.json"

if (-not (Test-Path $BatchScript)) {
    Write-Host "找不到文件: $BatchScript" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $GraphPath)) {
    Write-Host "找不到文件: $GraphPath(这是最上游的输入,不是流水线自己产出的,需要先准备好)" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "run_multi_paper_step1a_batch.py (paper_id=$PaperId)" -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan

python $BatchScript $PaperId

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "流水线失败,退出码 $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

# --- 跑完之后打印最终结果摘要 ---
Write-Host ""
Write-Host "===================================================================" -ForegroundColor Green
Write-Host "全部完成,结果摘要:" -ForegroundColor Green
Write-Host "===================================================================" -ForegroundColor Green

$ClaimsFinalPath = Join-Path $ScriptDir "data\$PaperId\claims_final.json"
if (Test-Path $ClaimsFinalPath) {
    $claimsFinal = Get-Content -Path $ClaimsFinalPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $nClaim = @($claimsFinal.claim).Count
    $nNote = @($claimsFinal.note).Count
    $nRelation = @($claimsFinal.relation).Count
    Write-Host "claim: $nClaim 条, note: $nNote 条, relation: $nRelation 条"
    Write-Host "完整结果: $ClaimsFinalPath"
} else {
    Write-Host "没找到 $ClaimsFinalPath——流水线可能中途停在了某一步,往上翻输出看在哪一步停的。" -ForegroundColor Yellow
}
