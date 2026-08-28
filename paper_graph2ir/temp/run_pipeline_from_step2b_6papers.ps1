# run_pipeline_from_step2b_6papers.ps1
#
# 放在 paper_graph2ir 项目根目录下(跟 claude_api_call.py、
# run_pipeline_from_step2b.py 同级)。
#
# 对6篇论文依次跑 run_pipeline_from_step2b.py(从step2b开始,跳过
# step1a/step1b/step2a)。单篇失败不会中断其余论文。
#
# 前提: 每篇论文的 organized_content.json 和 claim_completeness_analysis
# .json(带着step2a已经写好的字段)必须已经存在——这两个不是这个脚本
# 产出的;缺了的话,对应那一篇的step2b自己会报"找不到文件"、判定失败,
# 不会拖累其它论文。
#
# 用法:
#   .\run_pipeline_from_step2b_6papers.ps1

$ErrorActionPreference = "Stop"

if (-not $env:ANTHROPIC_API_KEY) {
    Write-Host "没检测到 ANTHROPIC_API_KEY 环境变量,先设置好再跑。" -ForegroundColor Red
    exit 1
}

$ScriptDir = $PSScriptRoot
$Script = Join-Path $ScriptDir "run_pipeline_from_step2b.py"

if (-not (Test-Path $Script)) {
    Write-Host "找不到文件: $Script" -ForegroundColor Red
    exit 1
}

$PaperIds = @(
    "867752822639165809",
    "1032903864883347458",
    "867750889056633390",
    "867757662605934651",
    "867760083600146646",
    "867771291879342656"
)

$Failed = @()

foreach ($id in $PaperIds) {
    Write-Host ""
    Write-Host "###################################################################" -ForegroundColor Magenta
    Write-Host "论文 $id" -ForegroundColor Magenta
    Write-Host "###################################################################" -ForegroundColor Magenta

    python $Script $id

    if ($LASTEXITCODE -ne 0) {
        Write-Host "${id} 失败,退出码 $LASTEXITCODE,继续跑下一篇" -ForegroundColor Red
        $Failed += $id
    }
}

# --- 6篇论文汇总 ---
Write-Host ""
Write-Host "===================================================================" -ForegroundColor Green
Write-Host "全部完成,6篇论文汇总:" -ForegroundColor Green
Write-Host "===================================================================" -ForegroundColor Green

foreach ($id in $PaperIds) {
    $ClaimsFinalPath = Join-Path $ScriptDir "data\$id\claims_final.json"
    if (-not (Test-Path $ClaimsFinalPath)) {
        Write-Host "  ${id}: 没有生成 claims_final.json" -ForegroundColor Yellow
        continue
    }
    $cf = Get-Content -Path $ClaimsFinalPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $nClaim = @($cf.claim).Count
    $nNote = @($cf.note).Count
    $nRelation = @($cf.relation).Count
    Write-Host "  ${id}: claim=$nClaim, note=$nNote, relation=$nRelation"
}

if ($Failed.Count -gt 0) {
    Write-Host ""
    Write-Host "以下论文没跑成功: $($Failed -join ', ')" -ForegroundColor Red
}
