# run_step2b_test_fix.ps1
#
# 放在项目根目录下。
#
# 单独重跑一次 step2b_complete_claims.py,验证prompt里新加的那条
# "supports/illustrates不算缺失内容"规则,能不能让conclusion_1的
# claim[8]/claim[10]不再互相吞并对方的实质内容。
#
# 注意:这会重新生成这篇论文全部20条claim的"完整表述"(不只是conclusion_1
# 这2条),而且只更新到claim_completeness_analysis.json这一份文件里——
# 更下游的claim_completeness_analysis.json(step2c/d)、step2_output_claims
# .json(step2e)、以及再往后的instance/relation相关文件,都是基于"旧的"
# 完整表述生成的,这次重跑不会自动帮你把它们也刷新。如果这次验证确认
# 修好了、你想让整篇论文完全反映这个修复,需要接着重跑step2c开始的
# 后续步骤。
#
# 用法:
#   .\run_step2b_test_fix.ps1

$ErrorActionPreference = "Stop"

$PaperId = "867771291879342656"

if (-not $env:ANTHROPIC_API_KEY) {
    Write-Host "没检测到 ANTHROPIC_API_KEY 环境变量,先设置好再跑。" -ForegroundColor Red
    exit 1
}

$ScriptDir = $PSScriptRoot
$Step2bScript = Join-Path $ScriptDir "step2_claim_completeness\step2b_complete_claims.py"
$CompletenessPath = Join-Path $ScriptDir "data\$PaperId\claim_completeness_analysis.json"

if (-not (Test-Path $Step2bScript)) {
    Write-Host "找不到文件: $Step2bScript" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $CompletenessPath)) {
    Write-Host "找不到文件: $CompletenessPath" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "step2b_complete_claims.py (paper_id=$PaperId,验证prompt修复)" -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan

python $Step2bScript $PaperId

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "失败,退出码 $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

# --- 重点看conclusion_1的claim[8]和claim[10] ---
Write-Host ""
Write-Host "===================================================================" -ForegroundColor Green
Write-Host "重点核对: conclusion_1 claim[8] / claim[10]" -ForegroundColor Green
Write-Host "===================================================================" -ForegroundColor Green

$records = Get-Content -Path $CompletenessPath -Raw -Encoding UTF8 | ConvertFrom-Json
$claim8 = $records | Where-Object { $_.conclusion -eq "conclusion_1" -and $_.number -eq 8 }
$claim10 = $records | Where-Object { $_.conclusion -eq "conclusion_1" -and $_.number -eq 10 }

Write-Host ""
Write-Host "claim[8](assertion)完整表述:" -ForegroundColor Yellow
Write-Host $claim8.完整表述
Write-Host ""
Write-Host "claim[10](example)完整表述:" -ForegroundColor Yellow
Write-Host $claim10.完整表述

Write-Host ""
if ($claim8.完整表述 -match "CIFAR-100 tickets generalize better" -or $claim10.完整表述 -match "class-rich source datasets produced stronger") {
    Write-Host "!! 看起来还是有一方把另一方的实质内容搬了过来,修复可能没生效或者不够。" -ForegroundColor Red
} else {
    Write-Host "两条claim看起来没有再互相搬运对方的实质内容。" -ForegroundColor Green
}

Write-Host ""
Write-Host "提醒: 这次只更新了 claim_completeness_analysis.json,step2c及之后的" -ForegroundColor Yellow
Write-Host "文件还是旧的,如果这次验证通过、想让整篇论文完全反映这个修复," -ForegroundColor Yellow
Write-Host "需要接着重新跑 step2c 开始的后续步骤。" -ForegroundColor Yellow
