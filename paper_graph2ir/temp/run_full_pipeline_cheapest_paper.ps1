# run_full_pipeline_cheapest_paper.ps1
#
# 放在 paper_graph2ir 项目根目录下(跟 claude_api_call.py、run_step1a.ps1 同级)。
# 对指定的一篇论文,依次跑完整的两步流程:
#   1. step1a_pipeline.py —— 重新调用API打标签,生成 conclusions_labeled.json
#   2. step1b_organize_labeled_content.py —— 整理成 organized_content.json
#
# 默认处理的是 paper_id = 867757662605934651。
# 挑这一篇的原因:在之前已经处理过的20篇论文里,这篇的 conclusion 数量最少
# (只有4条,第二少的都有5条)。因为每条 conclusion 都要把完整的 prompt
# 模板(将近19000字符)重新发一遍,conclusion 数量越少、调用API的次数就
# 越少,是这20篇里用新代码重新处理一遍花费最低的一篇。
#
# 用法:
#   .\run_full_pipeline_cheapest_paper.ps1                          # 用默认的这篇
#   .\run_full_pipeline_cheapest_paper.ps1 -PaperId "其它paper_id"   # 处理别的论文
#   .\run_full_pipeline_cheapest_paper.ps1 -NoBackup                # 不备份旧数据,直接覆盖
#
# 默认会先把这篇论文现有的 conclusions_labeled.json / organized_content.json
# (旧schema跑出来的)备份一份(加上时间戳后缀)再覆盖,方便需要的话对比新旧结果。

param(
    [string]$PaperId = "867757662605934651",
    [switch]$NoBackup
)

$ErrorActionPreference = "Stop"

if (-not $env:ANTHROPIC_API_KEY) {
    Write-Host "没检测到 ANTHROPIC_API_KEY 环境变量,先设置好再跑。" -ForegroundColor Red
    exit 1
}

$ScriptDir = $PSScriptRoot
$Step1aScript = Join-Path $ScriptDir "step1_process_conclusions\step1a_pipeline.py"
$Step1bScript = Join-Path $ScriptDir "step1_process_conclusions\step1b_organize_labeled_content.py"
$DataDir = Join-Path $ScriptDir "data\$PaperId"
$LabeledPath = Join-Path $DataDir "conclusions_labeled.json"
$OrganizedPath = Join-Path $DataDir "organized_content.json"

foreach ($p in @($Step1aScript, $Step1bScript)) {
    if (-not (Test-Path $p)) {
        Write-Host "找不到文件: $p" -ForegroundColor Red
        exit 1
    }
}

if (-not (Test-Path $DataDir)) {
    Write-Host "找不到论文目录: $DataDir" -ForegroundColor Red
    exit 1
}

# --- 备份旧数据(这篇论文之前是用旧schema跑出来的,默认先备份再覆盖) ---
if (-not $NoBackup) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    foreach ($f in @($LabeledPath, $OrganizedPath)) {
        if (Test-Path $f) {
            $bak = "$f.bak_$stamp"
            Copy-Item -Path $f -Destination $bak
            Write-Host "已备份旧文件: $bak" -ForegroundColor DarkGray
        }
    }
}

Write-Host ""
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "第一步: step1a_pipeline.py (重新调用API打标签, paper_id=$PaperId)" -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan
python $Step1aScript $PaperId
if ($LASTEXITCODE -ne 0) {
    Write-Host "step1a_pipeline.py 失败,退出码 $LASTEXITCODE" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "第二步: step1b_organize_labeled_content.py (整理成 organized_content.json)" -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan
python $Step1bScript $PaperId
if ($LASTEXITCODE -ne 0) {
    Write-Host "step1b_organize_labeled_content.py 失败,退出码 $LASTEXITCODE" -ForegroundColor Red
    exit 1
}

# --- 跑完之后打印一个简单的结果摘要 ---
Write-Host ""
Write-Host "===================================================================" -ForegroundColor Green
Write-Host "全部完成,结果摘要:" -ForegroundColor Green
Write-Host "===================================================================" -ForegroundColor Green

$labeled = Get-Content -Path $LabeledPath -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($c in $labeled) {
    $name = ($c.id -split "::")[-1]
    Write-Host "  $name -> labeling_status: $($c.labeling_status)"
}

$organized = Get-Content -Path $OrganizedPath -Raw -Encoding UTF8 | ConvertFrom-Json
$totalParts = ($organized | ForEach-Object { $_.organized_parts.Count } | Measure-Object -Sum).Sum
Write-Host ""
Write-Host "conclusions_labeled.json: $LabeledPath"
Write-Host "organized_content.json:   $OrganizedPath ($($organized.Count) 条conclusion, 共 $totalParts 个part)"
