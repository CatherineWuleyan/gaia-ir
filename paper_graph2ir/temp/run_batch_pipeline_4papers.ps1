# run_batch_pipeline_4papers.ps1
#
# 放在 paper_graph2ir 项目根目录下(跟 claude_api_call.py 同级)。
# 用 Batch API 一次性跑完这4篇论文的 step1a 打标签,再依次对每篇跑
# step1b 整理成 organized_content.json。
#
# 跟同步版本(run_full_pipeline_cheapest_paper.ps1)的区别:
#   - step1a 这一步改成调 step1a_batch_pipeline.py,内部会把重试的三档
#     参数拆成三"轮"batch提交,所有用量打五折,但是异步的——脚本会一直
#     等(内部自己轮询),官方说大多数一小时内完成,最多24小时,所以这次
#     跑完可能要等一阵子,不是提交完立刻出结果。
#   - 可以放心中途 Ctrl+C:进度存在 step1_process_conclusions\batch_state.json
#     里,重新跑这个脚本会接着上次的进度继续,不会重新提交、不会重复扣钱。
#   - step1b 这一步不受影响,还是跟以前一样逐篇同步跑。
#
# 默认处理这4篇:
#   1032903864883347458 / 867750889056633390 / 867752822639165809 / 867760083600146646
#
# 用法:
#   .\run_batch_pipeline_4papers.ps1
#   .\run_batch_pipeline_4papers.ps1 -PaperIds "id1","id2"   # 处理别的论文列表
#   .\run_batch_pipeline_4papers.ps1 -NoBackup               # 不备份旧数据,直接覆盖

param(
    [string[]]$PaperIds = @(
        "1032903864883347458",
        "867750889056633390",
        "867752822639165809",
        "867760083600146646"
    ),
    [switch]$NoBackup
)

$ErrorActionPreference = "Stop"

if (-not $env:ANTHROPIC_API_KEY) {
    Write-Host "没检测到 ANTHROPIC_API_KEY 环境变量,先设置好再跑。" -ForegroundColor Red
    exit 1
}

$ScriptDir = $PSScriptRoot
$Step1aBatchScript = Join-Path $ScriptDir "step1_process_conclusions\step1a_batch_pipeline.py"
$Step1bScript = Join-Path $ScriptDir "step1_process_conclusions\step1b_organize_labeled_content.py"

foreach ($p in @($Step1aBatchScript, $Step1bScript)) {
    if (-not (Test-Path $p)) {
        Write-Host "找不到文件: $p" -ForegroundColor Red
        exit 1
    }
}

# --- 备份旧数据 ---
if (-not $NoBackup) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    foreach ($paperId in $PaperIds) {
        $dataDir = Join-Path $ScriptDir "data\$paperId"
        foreach ($name in @("conclusions_labeled.json", "organized_content.json")) {
            $f = Join-Path $dataDir $name
            if (Test-Path $f) {
                $bak = "$f.bak_$stamp"
                Copy-Item -Path $f -Destination $bak
                Write-Host "已备份旧文件: $bak" -ForegroundColor DarkGray
            }
        }
    }
}

Write-Host ""
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "第一步: step1a_batch_pipeline.py (Batch API,处理 $($PaperIds.Count) 篇论文)" -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "注意: 这一步是异步的,官方说大多数batch一小时内完成、最多24小时," -ForegroundColor DarkYellow
Write-Host "      脚本会自己一直等,不需要你做任何操作。可以放心中途Ctrl+C," -ForegroundColor DarkYellow
Write-Host "      进度存在 step1_process_conclusions\batch_state.json 里," -ForegroundColor DarkYellow
Write-Host "      重新跑这个脚本会接着上次的进度继续,不会重新提交。" -ForegroundColor DarkYellow
Write-Host ""

python $Step1aBatchScript @PaperIds
if ($LASTEXITCODE -ne 0) {
    Write-Host "step1a_batch_pipeline.py 失败,退出码 $LASTEXITCODE" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "第二步: 依次对每篇论文跑 step1b_organize_labeled_content.py" -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan

foreach ($paperId in $PaperIds) {
    Write-Host ""
    Write-Host "--- $paperId ---" -ForegroundColor Cyan
    python $Step1bScript $paperId
    if ($LASTEXITCODE -ne 0) {
        Write-Host "step1b_organize_labeled_content.py 失败(paper_id=$paperId),退出码 $LASTEXITCODE" -ForegroundColor Red
        exit 1
    }
}

# --- 跑完之后打印结果摘要 ---
Write-Host ""
Write-Host "===================================================================" -ForegroundColor Green
Write-Host "全部完成,结果摘要:" -ForegroundColor Green
Write-Host "===================================================================" -ForegroundColor Green

foreach ($paperId in $PaperIds) {
    $dataDir = Join-Path $ScriptDir "data\$paperId"
    $labeledPath = Join-Path $dataDir "conclusions_labeled.json"
    $organizedPath = Join-Path $dataDir "organized_content.json"

    Write-Host ""
    Write-Host "$paperId" -ForegroundColor Cyan
    $labeled = Get-Content -Path $labeledPath -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($c in $labeled) {
        $name = ($c.id -split "::")[-1]
        Write-Host "  $name -> labeling_status: $($c.labeling_status)"
    }
    $organized = Get-Content -Path $organizedPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $totalParts = ($organized | ForEach-Object { $_.organized_parts.Count } | Measure-Object -Sum).Sum
    Write-Host "  organized_content.json: $($organized.Count) 条conclusion, 共 $totalParts 个part"
}
