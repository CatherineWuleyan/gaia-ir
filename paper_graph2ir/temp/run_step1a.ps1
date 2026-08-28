# run_step1a.ps1
# 放在 paper_graph2ir 项目根目录下(跟 claude_api_call.py 同级)。
# 调用 step1_process_conclusions\step1a_pipeline.py 处理指定 paper_id。
#
# 用法:
#   .\run_step1a.ps1                          # 不传参数,默认处理这篇论文
#   .\run_step1a.ps1 -PaperId "其它paper_id"   # 处理别的论文

param(
    [string]$PaperId = "867771291879342656"
)

$ScriptDir = $PSScriptRoot
$PipelineScript = Join-Path $ScriptDir "step1_process_conclusions\step1a_pipeline.py"

if (-not (Test-Path $PipelineScript)) {
    Write-Host "找不到文件: $PipelineScript" -ForegroundColor Red
    exit 1
}

python $PipelineScript $PaperId
