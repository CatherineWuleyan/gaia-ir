# run_step4c_review_example_relations.ps1
#
# 放在 paper_graph2ir 项目根目录下(跟 claude_api_call.py、
# run_step4a_step4b_5papers.ps1 同级)。
#
# 真正调用API跑 step4c_review_example_relations.py——审查论据/example
# 关系,对每条至少有一个target已经有实例化版本的关系,调 Sonnet 5(思考
# 深度high,三档模型/兜底机制沿用step2d)判断该连一般版本还是实例化版本
# 更直接,结果原地更新到 claims_final.json 的relation列表里。
#
# 默认处理 867757662605934651——这篇的 conclusion_3 里claim[4](general)
# 和claim[14](instantiated,替换了architectures/datasets/Trojan types
# 三个instance)之前已经生成好了,claim[5]/claim[6]各自有一条论据关系
# 指向claim[4],适合用来看这一步的实际判断效果。
#
# 前提:这篇论文要已经跑过 step4a + step4b(claims_final.json里要已经有
# instantiated claim和对应的"...的instance"关系,不然没有可审查的对象,
# 脚本会提示"没有需要审查的关系"然后什么都不做,不会报错)。
#
# 用法:
#   .\run_step4c_review_example_relations.ps1                        # 用默认的这篇
#   .\run_step4c_review_example_relations.ps1 -PaperId "其它paper_id"

param(
    [string]$PaperId = "867757662605934651"
)

$ErrorActionPreference = "Stop"

if (-not $env:ANTHROPIC_API_KEY) {
    Write-Host "没检测到 ANTHROPIC_API_KEY 环境变量,先设置好再跑。" -ForegroundColor Red
    exit 1
}

$ScriptDir = $PSScriptRoot
$Step4cScript = Join-Path $ScriptDir "step4_relation_construction\step4c_review_example_relations.py"
$DataDir = Join-Path $ScriptDir "data\$PaperId"

if (-not (Test-Path $Step4cScript)) {
    Write-Host "找不到文件: $Step4cScript" -ForegroundColor Red
    exit 1
}

$ClaimsFinalPath = Join-Path $DataDir "claims_final.json"
$OrganizedPath = Join-Path $DataDir "organized_content.json"
foreach ($p in @($ClaimsFinalPath, $OrganizedPath)) {
    if (-not (Test-Path $p)) {
        Write-Host "找不到文件: $p" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "step4c_review_example_relations.py (真实调用API, paper_id=$PaperId)" -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan

python $Step4cScript $PaperId

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "step4c_review_example_relations.py 失败,退出码 $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

# --- 跑完之后打印relation部分的最终结果,方便肉眼核对 ---
Write-Host ""
Write-Host "===================================================================" -ForegroundColor Green
Write-Host "relation部分最终结果:" -ForegroundColor Green
Write-Host "===================================================================" -ForegroundColor Green

$claimsFinal = Get-Content -Path $ClaimsFinalPath -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($r in $claimsFinal.relation) {
    Write-Host "  [$($r.conclusion)] $($r.expression)"
}
