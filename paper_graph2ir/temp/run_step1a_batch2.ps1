# run_step1a_batch2.ps1
# 放在 paper_graph2ir 项目根目录下(跟 claude_api_call.py 同级)。
# 依次用 step1a_pipeline.py 处理这三篇论文。

$ScriptDir = $PSScriptRoot
$PipelineScript = Join-Path $ScriptDir "step1_process_conclusions\step1a_pipeline.py"

if (-not (Test-Path $PipelineScript)) {
    Write-Host "找不到文件: $PipelineScript" -ForegroundColor Red
    exit 1
}

$paperIds = @(
    "924003605663449356",
    "867769419944689875",
    "867745350910214175"
)

foreach ($paperId in $paperIds) {
    Write-Host ""
    Write-Host "===================================================="
    Write-Host "开始处理 paper_id: $paperId"
    Write-Host "===================================================="
    python $PipelineScript $paperId
}
