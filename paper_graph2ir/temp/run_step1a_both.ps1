# run_step1a_both.ps1
# 放在 paper_graph2ir 项目根目录下(跟 claude_api_call.py 同级)。
# 依次用 step1a_pipeline.py 处理这两篇论文。

$ScriptDir = $PSScriptRoot
$PipelineScript = Join-Path $ScriptDir "step1_process_conclusions\step1a_pipeline.py"

if (-not (Test-Path $PipelineScript)) {
    Write-Host "找不到文件: $PipelineScript" -ForegroundColor Red
    exit 1
}

$paperIds = @(
    "867771291879342656",
    "867751999540560103"
)

foreach ($paperId in $paperIds) {
    Write-Host ""
    Write-Host "===================================================="
    Write-Host "开始处理 paper_id: $paperId"
    Write-Host "===================================================="
    python $PipelineScript $paperId
}
