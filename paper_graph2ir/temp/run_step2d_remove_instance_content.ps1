# run_step2d_remove_instance_content.ps1
#
# 放在 paper_graph2ir 项目根目录下(跟 claude_api_call.py、
# run_step2c_instance_containment.ps1 同级)。
#
# 真正调用API跑 step2d_remove_instance_content.py——按conclusion批量处理
# data\<PaperId>\claim_completeness_analysis.json 里 instance_containment
# ==True 的候选claim,调 Sonnet 5(思考深度high)复核每一条是不是真的混入
# 了instance内容,是就剥离、不是就原样保留,结果写回同一个文件(追加
# 完整表述_不含instance / instance_containment_confirmed /
# instance_removal_status 三个字段)。
#
# 默认处理 867752822639165809——这篇论文的 conclusion_5 有3条真实候选
# (claim 6/11/13),适合用来看批量处理、以及"同一批里有的改、有的不改"
# 这种混合结果的实际效果。
#
# 用法:
#   .\run_step2d_remove_instance_content.ps1                        # 用默认的这篇
#   .\run_step2d_remove_instance_content.ps1 -PaperId "其它paper_id"

param(
    [string]$PaperId = "867752822639165809"
)

$ErrorActionPreference = "Stop"

if (-not $env:ANTHROPIC_API_KEY) {
    Write-Host "没检测到 ANTHROPIC_API_KEY 环境变量,先设置好再跑。" -ForegroundColor Red
    exit 1
}

$ScriptDir = $PSScriptRoot
$Step2dScript = Join-Path $ScriptDir "step2_claim_completeness\step2d_remove_instance_content.py"
$DataDir = Join-Path $ScriptDir "data\$PaperId"
$ClaimAnalysisPath = Join-Path $DataDir "claim_completeness_analysis.json"
$OrganizedPath = Join-Path $DataDir "organized_content.json"

if (-not (Test-Path $Step2dScript)) {
    Write-Host "找不到文件: $Step2dScript" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $OrganizedPath)) {
    Write-Host "找不到文件: $OrganizedPath(需要先跑完 step1b_organize_labeled_content.py)" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $ClaimAnalysisPath)) {
    Write-Host "找不到文件: $ClaimAnalysisPath(需要先跑完 step2a/step2b/step2c)" -ForegroundColor Red
    exit 1
}

$existing = Get-Content -Path $ClaimAnalysisPath -Raw -Encoding UTF8 | ConvertFrom-Json
$hasContainmentField = $existing | Where-Object { $_.PSObject.Properties.Name -contains "instance_containment" } | Select-Object -First 1
if (-not $hasContainmentField) {
    Write-Host "$ClaimAnalysisPath 里还没有 instance_containment 字段(需要先跑完 step2c_check_instance_containment.py)" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "step2d_remove_instance_content.py (真实调用API, paper_id=$PaperId)" -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan

python $Step2dScript $PaperId

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "step2d_remove_instance_content.py 失败,退出码 $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

# --- 跑完之后打印结果摘要 ---
Write-Host ""
Write-Host "===================================================================" -ForegroundColor Green
Write-Host "全部完成,结果摘要:" -ForegroundColor Green
Write-Host "===================================================================" -ForegroundColor Green

$records = Get-Content -Path $ClaimAnalysisPath -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($r in $records) {
    if ($null -eq $r.instance_containment_confirmed) {
        continue  # 跳过的claim(instance_containment不是True)不列出来,避免刷屏
    }
    $marker = if ($r.instance_containment_confirmed -eq $true) { "确认混入,已剥离" } else { "复核后判定不是混入" }
    Write-Host ""
    Write-Host "  $($r.conclusion) claim[$($r.number)] -> $marker ($($r.instance_removal_status))"
    if ($r.instance_containment_confirmed -eq $true) {
        Write-Host "    改写前: $($r.完整表述)" -ForegroundColor DarkGray
        Write-Host "    改写后: $($r.完整表述_不含instance)" -ForegroundColor DarkGray
    }
}

$confirmed = @($records | Where-Object { $_.instance_containment_confirmed -eq $true }).Count
$rejected = @($records | Where-Object { $_.instance_containment -eq $true -and $_.instance_containment_confirmed -eq $false }).Count
$fallback = @($records | Where-Object { $_.instance_removal_status -eq "fallback_unchanged" }).Count

Write-Host ""
Write-Host "合计: $confirmed 条确认混入并剥离,$rejected 条复核后判定不是混入,$fallback 条走了保守兜底"
Write-Host "结果文件: $ClaimAnalysisPath"
