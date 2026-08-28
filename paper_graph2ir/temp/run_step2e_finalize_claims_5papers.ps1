# run_step2e_finalize_claims_5papers.ps1
#
# 放在 paper_graph2ir 项目根目录下(跟 claude_api_call.py、
# run_step2d_remove_instance_content.ps1 同级)。
#
# step2e_remove_instance_content.py 不调API,纯本地计算,所以这个脚本
# 不用检查 ANTHROPIC_API_KEY,直接把5篇论文一次跑完。前提是每篇论文都
# 已经跑过 step2c + step2d(claim_completeness_analysis.json里得有
# "完整表述_不含instance"和"instance_containment_confirmed"这两个字段,
# 缺了会被跳过并提示)。
#
# 用法:
#   .\run_step2e_finalize_claims_5papers.ps1

$ErrorActionPreference = "Stop"

$ScriptDir = $PSScriptRoot
$Step2eScript = Join-Path $ScriptDir "step2_claim_completeness\step2e_finalize_claims.py"

if (-not (Test-Path $Step2eScript)) {
    Write-Host "找不到文件: $Step2eScript" -ForegroundColor Red
    exit 1
}

$PaperIds = @(
    "867752822639165809",
    "1032903864883347458",
    "867750889056633390",
    "867757662605934651",
    "867760083600146646"
)

$Failed = @()
$TotalClaims = 0
$TotalRemoved = 0
$TotalPure = 0

foreach ($id in $PaperIds) {
    $ClaimAnalysisPath = Join-Path $ScriptDir "data\$id\claim_completeness_analysis.json"

    if (-not (Test-Path $ClaimAnalysisPath)) {
        Write-Host "跳过 ${id}: 找不到 $ClaimAnalysisPath" -ForegroundColor Yellow
        $Failed += $id
        continue
    }

    Write-Host ""
    Write-Host "===================================================================" -ForegroundColor Cyan
    Write-Host "step2e_finalize_claims.py (paper_id=$id)" -ForegroundColor Cyan
    Write-Host "===================================================================" -ForegroundColor Cyan

    python $Step2eScript $id

    if ($LASTEXITCODE -ne 0) {
        Write-Host "${id} 失败,退出码 $LASTEXITCODE,继续跑下一篇" -ForegroundColor Red
        $Failed += $id
        continue
    }

    $OutPath = Join-Path $ScriptDir "data\$id\step2_output_claims.json"
    if (Test-Path $OutPath) {
        $claims = Get-Content -Path $OutPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $n = @($claims).Count
        $nRemoved = @($claims | Where-Object { $_.instance_removed -eq $true }).Count
        $nPure = @($claims | Where-Object { $_.is_pure_data -eq $true }).Count

        $TotalClaims += $n
        $TotalRemoved += $nRemoved
        $TotalPure += $nPure
    }
}

# --- 5篇论文汇总统计 ---
Write-Host ""
Write-Host "===================================================================" -ForegroundColor Green
Write-Host "全部完成,汇总:" -ForegroundColor Green
Write-Host "===================================================================" -ForegroundColor Green
Write-Host "合计: $TotalClaims 条claim,$TotalRemoved 条剥离了instance内容,$TotalPure 条是纯实验数据"

if ($Failed.Count -gt 0) {
    Write-Host ""
    Write-Host "以下论文没跑成功: $($Failed -join ', ')" -ForegroundColor Red
}
