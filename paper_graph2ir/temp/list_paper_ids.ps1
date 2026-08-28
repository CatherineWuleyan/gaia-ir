# list_paper_ids.ps1
# 放在 paper_graph2ir 项目根目录下(跟 data\ 同级)。
# 把 data\ 目录下所有子文件夹的名字(也就是所有 paper_id)写进 paper_ids.txt,一行一个。

$ScriptDir = $PSScriptRoot
$DataDir = Join-Path $ScriptDir "data"
$OutFile = Join-Path $ScriptDir "paper_ids.txt"

if (-not (Test-Path $DataDir)) {
    Write-Host "找不到目录: $DataDir" -ForegroundColor Red
    exit 1
}

$paperIds = Get-ChildItem -Path $DataDir -Directory | Select-Object -ExpandProperty Name

$paperIds | Out-File -FilePath $OutFile -Encoding utf8

Write-Host "共找到 $($paperIds.Count) 个 paper_id,已写入 $OutFile"
