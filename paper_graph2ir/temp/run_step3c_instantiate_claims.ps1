# run_step3c_instantiate_claims.ps1
#
# 放在 paper_graph2ir 项目根目录下(跟 claude_api_call.py、
# run_step2d_remove_instance_content.ps1 同级)。
#
# 真正调用API跑 step3c_instantiate_claims.py——按conclusion批量处理
# data\<PaperId>\claim_instance_associations.json 里certain_instances/
# uncertain_instances非空的待处理claim,调 Sonnet 5(思考深度high,三档
# 模型/兜底机制完全沿用step2d)把claim里被instance举例说明的一般性词语
# 换成instance给出的具体内容,结果写到 data\<PaperId>\special_claims.json
# (只包含真的发生了改写的claim)。
#
# 依赖(缺一个都跑不了,脚本会先检查):
#   organized_content.json / instance_target_positions.json(step3a) /
#   claim_instance_associations.json(step3b) / step2_output_claims.json
#   (step2e)
#
# 默认处理 867757662605934651——这篇的 conclusion_3 claim[1] 一条claim
# 同时关联了3个instance(architectures/datasets/Trojan types),是目前
# 数据里最复杂的一个真实案例,适合用来看效果。
#
# 用法:
#   .\run_step3c_instantiate_claims.ps1                        # 用默认的这篇
#   .\run_step3c_instantiate_claims.ps1 -PaperId "其它paper_id"

param(
    [string]$PaperId = "867757662605934651"
)

$ErrorActionPreference = "Stop"

if (-not $env:ANTHROPIC_API_KEY) {
    Write-Host "没检测到 ANTHROPIC_API_KEY 环境变量,先设置好再跑。" -ForegroundColor Red
    exit 1
}

$ScriptDir = $PSScriptRoot
$Step3cScript = Join-Path $ScriptDir "step3_instance_extraction\step3c_instantiate_claims.py"
$DataDir = Join-Path $ScriptDir "data\$PaperId"

if (-not (Test-Path $Step3cScript)) {
    Write-Host "找不到文件: $Step3cScript" -ForegroundColor Red
    exit 1
}

$RequiredFiles = @(
    "organized_content.json",
    "instance_target_positions.json",
    "claim_instance_associations.json",
    "step2_output_claims.json"
)
foreach ($f in $RequiredFiles) {
    $p = Join-Path $DataDir $f
    if (-not (Test-Path $p)) {
        Write-Host "找不到文件: $p" -ForegroundColor Red
        Write-Host "(organized_content.json缺=step1没跑;instance_target_positions.json缺=step3a没跑;" -ForegroundColor Yellow
        Write-Host " claim_instance_associations.json缺=step3b没跑;step2_output_claims.json缺=step2e没跑)" -ForegroundColor Yellow
        exit 1
    }
}

Write-Host ""
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "step3c_instantiate_claims.py (真实调用API, paper_id=$PaperId)" -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan

python $Step3cScript $PaperId

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "step3c_instantiate_claims.py 失败,退出码 $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

# --- 跑完之后打印结果摘要 ---
Write-Host ""
Write-Host "===================================================================" -ForegroundColor Green
Write-Host "全部完成,结果摘要:" -ForegroundColor Green
Write-Host "===================================================================" -ForegroundColor Green

$OutPath = Join-Path $DataDir "special_claims.json"
$specialClaims = Get-Content -Path $OutPath -Raw -Encoding UTF8 | ConvertFrom-Json

if (@($specialClaims).Count -eq 0) {
    Write-Host "  这篇论文没有特殊claim(special_claims.json是空列表)。"
} else {
    foreach ($c in $specialClaims) {
        Write-Host ""
        Write-Host "  $($c.conclusion) claim[$($c.claim_number)]" -ForegroundColor Yellow
        Write-Host "    确定性instance: $($c.certain_instances_given -join ', ')"
        Write-Host "    不确定性instance: $($c.uncertain_instances_given -join ', ')"
        Write-Host "    实例化后: $($c.text)"
    }
}

Write-Host ""
Write-Host "结果文件: $OutPath"
