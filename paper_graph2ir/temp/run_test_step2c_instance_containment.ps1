# run_test_step2c_instance_containment.ps1
#
# 放在 paper_graph2ir 项目根目录下(跟 claude_api_call.py、
# run_full_pipeline_cheapest_paper.ps1 同级)。
#
# 跑 step2_claim_completeness/test_step2c_check_instance_containment.py——
# 这个测试文件本身会 monkeypatch 掉 call_claude(),不会真的发起任何网络
# 请求,所以:
#   - 不需要设置 ANTHROPIC_API_KEY(测试全程不会真的调用API)
#   - 但 claude_api_call.py 顶层有 `import anthropic`,所以运行环境里还是
#     要装好 anthropic 这个包(`pip install anthropic`),否则连 import
#     都会失败,这跟"要不要真的调用API"是两回事
#
# 测试内容:用真实的 data\867752822639165809、data\867757662605934651 这两篇
# 论文已有的 organized_content.json / claim_completeness_analysis.json,
# 验证 conclusion_4 claim[20] 应该被判 True(它的完整表述确实吸收了
# instance[13]的举例内容)、纯数据claim会被跳过不调API、重试链和保守兜底
# (选True)三条路径都正确。
#
# 用法:
#   .\run_test_step2c_instance_containment.ps1

$ErrorActionPreference = "Stop"

$ScriptDir = $PSScriptRoot
$TestScript = Join-Path $ScriptDir "step2_claim_completeness\test_step2c_check_instance_containment.py"

if (-not (Test-Path $TestScript)) {
    Write-Host "找不到文件: $TestScript" -ForegroundColor Red
    exit 1
}

python -c "import anthropic" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "没检测到 anthropic 包(claude_api_call.py 顶层需要 import 它)。" -ForegroundColor Red
    Write-Host "先跑: pip install anthropic" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "跑 test_step2c_check_instance_containment.py(不会真的调用API)" -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan

python $TestScript

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "测试失败,退出码 $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "===================================================================" -ForegroundColor Green
Write-Host "全部测试通过" -ForegroundColor Green
Write-Host "===================================================================" -ForegroundColor Green
