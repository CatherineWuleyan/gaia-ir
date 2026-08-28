# run_step2c_instance_containment.ps1
#
# 放在 paper_graph2ir 项目根目录下(跟 claude_api_call.py、
# run_full_pipeline_cheapest_paper.ps1 同级)。
#
# 真正调用API跑 step2c_check_instance_containment.py——这一步会读
# data\<PaperId>\organized_content.json 和 claim_completeness_analysis.json,
# 对其中每条非纯数据、且所在conclusion有instance的claim,调Haiku(必要时
# 升级到Sonnet 5)判断它有没有包含某个instance的举例内容,结果写回
# claim_completeness_analysis.json(追加 instance_containment /
# instance_containment_status 两个字段)。
#
# 默认处理 867752822639165809——这篇论文的 conclusion_4 我们已经人工核实
# 过:claim[20]的完整表述确实吸收了instance[13]举的OOD scoring方法列表,
# 适合用来验真实API的判断跟人工核对的结论对不对得上。
#
# 用法:
#   .\run_step2c_instance_containment.ps1                        # 用默认的这篇
#   .\run_step2c_instance_containment.ps1 -PaperId "其它paper_id"

param(
    [string]$PaperId = "867752822639165809"
)

$ErrorActionPreference = "Stop"

if (-not $env:ANTHROPIC_API_KEY) {
    Write-Host "没检测到 ANTHROPIC_API_KEY 环境变量,先设置好再跑。" -ForegroundColor Red
    exit 1
}

$ScriptDir = $PSScriptRoot
$Step2cScript = Join-Path $ScriptDir "step2_claim_completeness\step2c_check_instance_containment.py"
$DataDir = Join-Path $ScriptDir "data\$PaperId"
$ClaimAnalysisPath = Join-Path $DataDir "claim_completeness_analysis.json"
$OrganizedPath = Join-Path $DataDir "organized_content.json"

if (-not (Test-Path $Step2cScript)) {
    Write-Host "找不到文件: $Step2cScript" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $OrganizedPath)) {
    Write-Host "找不到文件: $OrganizedPath(需要先跑完 step1b_organize_labeled_content.py)" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $ClaimAnalysisPath)) {
    Write-Host "找不到文件: $ClaimAnalysisPath(需要先跑完 step2a_check_pure_data.py / step2b_complete_claims.py)" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "step2c_check_instance_containment.py (真实调用API, paper_id=$PaperId)" -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan

python $Step2cScript $PaperId

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "step2c_check_instance_containment.py 失败,退出码 $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

# --- 跑完之后打印结果摘要 ---
Write-Host ""
Write-Host "===================================================================" -ForegroundColor Green
Write-Host "全部完成,结果摘要:" -ForegroundColor Green
Write-Host "===================================================================" -ForegroundColor Green

$records = Get-Content -Path $ClaimAnalysisPath -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($r in $records) {
    $mark = if ($r.instance_containment -eq $true) { "True" }
            elseif ($r.instance_containment -eq $false) { "False" }
            else { "null" }
    Write-Host "  $($r.conclusion) claim[$($r.number)] -> instance_containment=$mark ($($r.instance_containment_status))"
}

Write-Host ""
Write-Host "结果文件: $ClaimAnalysisPath"
