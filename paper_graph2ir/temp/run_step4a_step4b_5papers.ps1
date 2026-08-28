# run_step4a_step4b_5papers.ps1
#
# 放在 paper_graph2ir 项目根目录下(跟 claude_api_call.py 同级)。
#
# step4a_build_claims_final.py 和 step4b_add_instantiated_claims.py 都不调
# API,纯本地计算,所以不用检查 ANTHROPIC_API_KEY。一次跑完5篇论文的
# step4a再step4b,每篇论文先跑step4a(需要organized_content.json+
# step2_output_claims.json),再跑step4b(还需要special_claims.json,来自
# step3c——如果这篇论文的step3c跑出来是空列表[],step4b会正常处理成
# "0条追加",不会报错;真的没有这个文件才会被跳过)。
#
# 用法:
#   .\run_step4a_step4b_5papers.ps1

$ErrorActionPreference = "Stop"

$ScriptDir = $PSScriptRoot
$Step4aScript = Join-Path $ScriptDir "step4_relation_construction\step4a_build_claims_final.py"
$Step4bScript = Join-Path $ScriptDir "step4_relation_construction\step4b_add_instantiated_claims.py"

if (-not (Test-Path $Step4aScript)) {
    Write-Host "找不到文件: $Step4aScript" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $Step4bScript)) {
    Write-Host "找不到文件: $Step4bScript" -ForegroundColor Red
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

foreach ($id in $PaperIds) {
    $DataDir = Join-Path $ScriptDir "data\$id"

    Write-Host ""
    Write-Host "===================================================================" -ForegroundColor Cyan
    Write-Host "step4a_build_claims_final.py (paper_id=$id)" -ForegroundColor Cyan
    Write-Host "===================================================================" -ForegroundColor Cyan

    $OrganizedPath = Join-Path $DataDir "organized_content.json"
    $Step2OutputPath = Join-Path $DataDir "step2_output_claims.json"
    if (-not (Test-Path $OrganizedPath) -or -not (Test-Path $Step2OutputPath)) {
        Write-Host "跳过 ${id}: 缺少 organized_content.json 或 step2_output_claims.json" -ForegroundColor Yellow
        $Failed += $id
        continue
    }

    python $Step4aScript $id
    if ($LASTEXITCODE -ne 0) {
        Write-Host "${id} 的step4a失败,退出码 $LASTEXITCODE,跳过这篇的step4b" -ForegroundColor Red
        $Failed += $id
        continue
    }

    Write-Host ""
    Write-Host "step4b_add_instantiated_claims.py (paper_id=$id)" -ForegroundColor Cyan

    $SpecialClaimsPath = Join-Path $DataDir "special_claims.json"
    if (-not (Test-Path $SpecialClaimsPath)) {
        Write-Host "跳过 ${id} 的step4b: 找不到 $SpecialClaimsPath(需要先跑step3c)" -ForegroundColor Yellow
        $Failed += $id
        continue
    }

    python $Step4bScript $id
    if ($LASTEXITCODE -ne 0) {
        Write-Host "${id} 的step4b失败,退出码 $LASTEXITCODE" -ForegroundColor Red
        $Failed += $id
    }
}

# --- 5篇论文汇总 ---
Write-Host ""
Write-Host "===================================================================" -ForegroundColor Green
Write-Host "全部完成,汇总:" -ForegroundColor Green
Write-Host "===================================================================" -ForegroundColor Green

foreach ($id in $PaperIds) {
    $ClaimsFinalPath = Join-Path $ScriptDir "data\$id\claims_final.json"
    if (-not (Test-Path $ClaimsFinalPath)) {
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
    Write-Host "以下论文有步骤没跑成功: $($Failed -join ', ')" -ForegroundColor Red
}
