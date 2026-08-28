# run_step1a_batch20.ps1
# 放在 paper_graph2ir 项目根目录下(跟 claude_api_call.py 同级)。
# 依次用 step1a_pipeline.py 处理这20篇论文(含之前已经处理过的5篇)。

$ScriptDir = $PSScriptRoot
$PipelineScript = Join-Path $ScriptDir "step1_process_conclusions\step1a_pipeline.py"

if (-not (Test-Path $PipelineScript)) {
    Write-Host "找不到文件: $PipelineScript" -ForegroundColor Red
    exit 1
}

$paperIds = @(
    "867771291879342656",
    "867751999540560103",
    "924003605663449356",
    "867769419944689875",
    "867745350910214175",
    "1095943058349883393",
    "867761916582298079",
    "867746711882170612",
    "867764414206443886",
    "1229506077859512414",
    "867758731482366166",
    "1223972462526464001",
    "867757662605934651",
    "873226422896820806",
    "867751811367305226",
    "867760083600146646",
    "1032903864883347458",
    "867757746668176351",
    "867750889056633390",
    "867752822639165809"
)

foreach ($paperId in $paperIds) {
    Write-Host ""
    Write-Host "===================================================="
    Write-Host "开始处理 paper_id: $paperId"
    Write-Host "===================================================="
    python $PipelineScript $paperId
}
