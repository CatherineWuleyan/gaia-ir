# run_step4c_review_example_relations_remaining4papers.ps1
#
# 放在 paper_graph2ir 项目根目录下(跟 claude_api_call.py、
# run_step4c_review_example_relations.ps1 同级)。
#
# 867757662605934651 已经手工跑过、人工核对过质量了,这个脚本跑剩下的
# 4篇论文:
#   867752822639165809
#   1032903864883347458
#   867750889056633390
#   867760083600146646
#
# 每篇论文调一次 step4c_review_example_relations.py(真实调用API)。前提
# 是这4篇都已经跑过 step4a + step4b(claims_final.json里要有instantiated
# claim和对应的"...的instance"关系,不然没有可审查的对象——step4c自己
# 遇到这种情况会打印"没有需要审查的关系"然后正常退出,不算失败,这个
# 脚本不会因此把这篇论文标记成失败)。单篇失败不会中断其余论文。
#
# 用法:
#   .\run_step4c_review_example_relations_remaining4papers.ps1

$ErrorActionPreference = "Stop"

if (-not $env:ANTHROPIC_API_KEY) {
    Write-Host "没检测到 ANTHROPIC_API_KEY 环境变量,先设置好再跑。" -ForegroundColor Red
    exit 1
}

$ScriptDir = $PSScriptRoot
$Step4cScript = Join-Path $ScriptDir "step4_relation_construction\step4c_review_example_relations.py"

if (-not (Test-Path $Step4cScript)) {
    Write-Host "找不到文件: $Step4cScript" -ForegroundColor Red
    exit 1
}

$RemainingPaperIds = @(
    "867752822639165809",
    "1032903864883347458",
    "867750889056633390",
    "867760083600146646"
)

$Failed = @()

foreach ($id in $RemainingPaperIds) {
    $DataDir = Join-Path $ScriptDir "data\$id"
    $ClaimsFinalPath = Join-Path $DataDir "claims_final.json"
    $OrganizedPath = Join-Path $DataDir "organized_content.json"

    if (-not (Test-Path $ClaimsFinalPath) -or -not (Test-Path $OrganizedPath)) {
        Write-Host "跳过 ${id}: 缺少 claims_final.json 或 organized_content.json(需要先跑step4a)" -ForegroundColor Yellow
        $Failed += $id
        continue
    }

    Write-Host ""
    Write-Host "===================================================================" -ForegroundColor Cyan
    Write-Host "step4c_review_example_relations.py (paper_id=$id)" -ForegroundColor Cyan
    Write-Host "===================================================================" -ForegroundColor Cyan

    python $Step4cScript $id

    if ($LASTEXITCODE -ne 0) {
        Write-Host "${id} 失败,退出码 $LASTEXITCODE,继续跑下一篇" -ForegroundColor Red
        $Failed += $id
    }
}

# --- 5篇论文(含之前已经跑过的867757662605934651)relation部分汇总 ---
Write-Host ""
Write-Host "===================================================================" -ForegroundColor Green
Write-Host "全部完成,5篇论文relation汇总:" -ForegroundColor Green
Write-Host "===================================================================" -ForegroundColor Green

$AllPaperIds = @("867757662605934651") + $RemainingPaperIds

foreach ($id in $AllPaperIds) {
    $ClaimsFinalPath = Join-Path $ScriptDir "data\$id\claims_final.json"
    if (-not (Test-Path $ClaimsFinalPath)) {
        continue
    }
    $claimsFinal = Get-Content -Path $ClaimsFinalPath -Raw -Encoding UTF8 | ConvertFrom-Json
    Write-Host ""
    Write-Host "  ${id}:"
    foreach ($r in $claimsFinal.relation) {
        Write-Host "    [$($r.conclusion)] $($r.expression)"
    }
}

if ($Failed.Count -gt 0) {
    Write-Host ""
    Write-Host "以下论文没跑成功: $($Failed -join ', ')" -ForegroundColor Red
}
