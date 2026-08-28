# run_step3a_step3b_5papers.ps1
#
# 放在 paper_graph2ir 项目根目录下(跟 claude_api_call.py、
# run_step3c_instantiate_claims.ps1 同级)。
#
# step3a_locate_instance_targets.py 和 step3b_classify_claim_instances.py
# 都不调API,纯本地计算,所以不用检查 ANTHROPIC_API_KEY。两个脚本都原生
# 支持 "all" 参数(自动处理 data\ 目录下全部论文,单篇论文缺文件会被
# 各自的process_paper跳过、打印提示,不会中断整个批处理),这里直接用
# 这个模式一次跑完全部论文,不用在PowerShell里手动列paper_id。
#
# 用法:
#   .\run_step3a_step3b_5papers.ps1

$ErrorActionPreference = "Stop"

$ScriptDir = $PSScriptRoot
$Step3aScript = Join-Path $ScriptDir "step3_instance_extraction\step3a_locate_instance_targets.py"
$Step3bScript = Join-Path $ScriptDir "step3_instance_extraction\step3b_classify_claim_instances.py"

if (-not (Test-Path $Step3aScript)) {
    Write-Host "找不到文件: $Step3aScript" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $Step3bScript)) {
    Write-Host "找不到文件: $Step3bScript" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "step3a_locate_instance_targets.py (all)" -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan

python $Step3aScript all

if ($LASTEXITCODE -ne 0) {
    Write-Host "step3a失败,退出码 $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "step3b_classify_claim_instances.py (all)" -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan

python $Step3bScript all

if ($LASTEXITCODE -ne 0) {
    Write-Host "step3b失败,退出码 $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "===================================================================" -ForegroundColor Green
Write-Host "全部完成" -ForegroundColor Green
Write-Host "===================================================================" -ForegroundColor Green
