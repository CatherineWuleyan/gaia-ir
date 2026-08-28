# run_step3c_instantiate_claims_5papers.ps1
#
# 放在 paper_graph2ir 项目根目录下(跟 claude_api_call.py、
# run_step3c_instantiate_claims.ps1 同级)。
#
# 真正调用API,把5篇论文的 step3c_instantiate_claims.py 一次跑完。每篇
# 论文跑之前会检查4个前置文件(organized_content.json / step3a产物
# instance_target_positions.json / step3b产物claim_instance_associations
# .json / step2e产物step2_output_claims.json),缺文件的论文会被跳过、
# 打印提示,不会中断其余论文。
#
# 用法:
#   .\run_step3c_instantiate_claims_5papers.ps1

$ErrorActionPreference = "Stop"

if (-not $env:ANTHROPIC_API_KEY) {
    Write-Host "没检测到 ANTHROPIC_API_KEY 环境变量,先设置好再跑。" -ForegroundColor Red
    exit 1
}

$ScriptDir = $PSScriptRoot
$Step3cScript = Join-Path $ScriptDir "step3_instance_extraction\step3c_instantiate_claims.py"

if (-not (Test-Path $Step3cScript)) {
    Write-Host "找不到文件: $Step3cScript" -ForegroundColor Red
    exit 1
}

$PaperIds = @(
    "867752822639165809",
    "1032903864883347458",
    "867750889056633390",
    "867757662605934651",
    "867760083600146646"
)

$RequiredFiles = @(
    "organized_content.json",
    "instance_target_positions.json",
    "claim_instance_associations.json",
    "step2_output_claims.json"
)

$Failed = @()
$TotalSpecial = 0

foreach ($id in $PaperIds) {
    $DataDir = Join-Path $ScriptDir "data\$id"

    $missing = $RequiredFiles | Where-Object { -not (Test-Path (Join-Path $DataDir $_)) }
    if ($missing.Count -gt 0) {
        Write-Host "跳过 ${id}: 缺少 $($missing -join ', ')" -ForegroundColor Yellow
        $Failed += $id
        continue
    }

    Write-Host ""
    Write-Host "===================================================================" -ForegroundColor Cyan
    Write-Host "step3c_instantiate_claims.py (paper_id=$id)" -ForegroundColor Cyan
    Write-Host "===================================================================" -ForegroundColor Cyan

    python $Step3cScript $id

    if ($LASTEXITCODE -ne 0) {
        Write-Host "${id} 失败,退出码 $LASTEXITCODE,继续跑下一篇" -ForegroundColor Red
        $Failed += $id
        continue
    }

    $OutPath = Join-Path $DataDir "special_claims.json"
    if (Test-Path $OutPath) {
        $specialClaims = Get-Content -Path $OutPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $TotalSpecial += @($specialClaims).Count
    }
}

# --- 5篇论文汇总 ---
Write-Host ""
Write-Host "===================================================================" -ForegroundColor Green
Write-Host "全部完成,汇总:" -ForegroundColor Green
Write-Host "===================================================================" -ForegroundColor Green

foreach ($id in $PaperIds) {
    $OutPath = Join-Path $ScriptDir "data\$id\special_claims.json"
    if (-not (Test-Path $OutPath)) {
        continue
    }
    $specialClaims = Get-Content -Path $OutPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $n = @($specialClaims).Count
    Write-Host "  ${id}: $n 条特殊claim"
}

Write-Host ""
Write-Host "合计: $TotalSpecial 条特殊claim"

if ($Failed.Count -gt 0) {
    Write-Host ""
    Write-Host "以下论文没跑成功: $($Failed -join ', ')" -ForegroundColor Red
}
