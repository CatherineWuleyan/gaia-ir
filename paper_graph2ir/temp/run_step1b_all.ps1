# run_step1b_all.ps1
# 放在 paper_graph2ir 项目根目录下(跟 paper_ids.txt、claude_api_call.py 同级)。
# 读取 paper_ids.txt 里的所有 paper_id,依次跑 step1b_organize_labeled_content.py。
# 没有 conclusions_labeled.json 的 paper_id 会被脚本自己跳过,不会中断这里的循环。

$ScriptDir = $PSScriptRoot
$PipelineScript = Join-Path $ScriptDir "step1_process_conclusions\step1b_organize_labeled_content.py"
$PaperIdsFile = Join-Path $ScriptDir "paper_ids.txt"

if (-not (Test-Path $PipelineScript)) {
    Write-Host "找不到文件: $PipelineScript" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $PaperIdsFile)) {
    Write-Host "找不到文件: $PaperIdsFile" -ForegroundColor Red
    exit 1
}

$paperIds = Get-Content -Path $PaperIdsFile | Where-Object { $_.Trim() -ne "" }

Write-Host "共读到 $($paperIds.Count) 个 paper_id"

foreach ($paperId in $paperIds) {
    Write-Host ""
    Write-Host "===================================================="
    Write-Host "开始处理 paper_id: $paperId"
    Write-Host "===================================================="
    python $PipelineScript $paperId
}
