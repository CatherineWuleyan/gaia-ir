# run_step4_fix_mirror_potential_paper.ps1
#
# 放在 paper_graph2ir 项目根目录下。
#
# 修复了step4a里"【1, 4】这种一个括号多个逗号分隔编号"解析不了的bug后,
# 重新跑这篇论文完整的step4(a->b->c->d)。之所以要重新跑全部4步、
# 不是只跑step4a:step4a是从organized_content.json+step2_output_claims
# .json重新整个构建claims_final.json的,会把之前step4b/c/d已经加进去的
# 内容全部覆盖掉,所以a跑完必须紧接着把b/c/d按顺序重新跑一遍,才能拿到
# 完整、正确的最终结果。organized_content.json和step2_output_claims
# .json都没有变,不需要重新跑更早的API调用步骤(step1a~step2e)。
#
# 用法:
#   .\run_step4_fix_mirror_potential_paper.ps1

$ErrorActionPreference = "Stop"

if (-not $env:ANTHROPIC_API_KEY) {
    Write-Host "没检测到 ANTHROPIC_API_KEY 环境变量,先设置好再跑(step4c需要调API)。" -ForegroundColor Red
    exit 1
}

$ScriptDir = $PSScriptRoot
$Step4Dir = Join-Path $ScriptDir "step4_relation_construction"

$Steps = @(
    @{ Name = "step4a_build_claims_final.py"; NeedsApi = $false },
    @{ Name = "step4b_add_instantiated_claims.py"; NeedsApi = $false },
    @{ Name = "step4c_review_example_relations.py"; NeedsApi = $true },
    @{ Name = "step4d_normalize_instance_relations.py"; NeedsApi = $false }
)

foreach ($step in $Steps) {
    $p = Join-Path $Step4Dir $step.Name
    if (-not (Test-Path $p)) {
        Write-Host "找不到文件: $p" -ForegroundColor Red
        exit 1
    }
}

$PaperIds = @("1032903864883347458")

$Failed = @()

foreach ($id in $PaperIds) {
    Write-Host ""
    Write-Host "###################################################################" -ForegroundColor Magenta
    Write-Host "论文 $id" -ForegroundColor Magenta
    Write-Host "###################################################################" -ForegroundColor Magenta

    $paperFailed = $false

    foreach ($step in $Steps) {
        $scriptPath = Join-Path $Step4Dir $step.Name
        $tag = if ($step.NeedsApi) { "[调API]" } else { "" }

        Write-Host ""
        Write-Host "=== $($step.Name) $tag ===" -ForegroundColor Cyan

        python $scriptPath $id

        if ($LASTEXITCODE -ne 0) {
            Write-Host "${id} 在 $($step.Name) 失败,退出码 $LASTEXITCODE,跳过这篇论文剩下的step4步骤" -ForegroundColor Red
            $Failed += $id
            $paperFailed = $true
            break
        }
    }

    if (-not $paperFailed) {
        Write-Host ""
        Write-Host "${id}: step4全部4步跑完" -ForegroundColor Green
    }
}

# --- 汇总 ---
Write-Host ""
Write-Host "===================================================================" -ForegroundColor Green
Write-Host "全部完成,结果汇总:" -ForegroundColor Green
Write-Host "===================================================================" -ForegroundColor Green

foreach ($id in $PaperIds) {
    $ClaimsFinalPath = Join-Path $ScriptDir "data\$id\claims_final.json"
    if (-not (Test-Path $ClaimsFinalPath)) {
        Write-Host "  ${id}: 没有生成 claims_final.json" -ForegroundColor Yellow
        continue
    }
    $cf = Get-Content -Path $ClaimsFinalPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $nClaim = @($cf.claim).Count
    $nNote = @($cf.note).Count
    $nRelation = @($cf.relation).Count
    Write-Host "  ${id}: claim=$nClaim, note=$nNote, relation=$nRelation"

    # 顺手检查一下还有没有残留的全角引用记号,验证这次的bug真的修好了
    $hasLeftover = $false
    foreach ($c in $cf.claim) {
        if ($c.text -match "【|】") { $hasLeftover = $true }
    }
    if ($hasLeftover) {
        Write-Host "    !! 还有claim里残留着未替换的全角引用记号,需要再查一下" -ForegroundColor Red
    } else {
        Write-Host "    没有发现残留的全角引用记号" -ForegroundColor Green
    }
}

if ($Failed.Count -gt 0) {
    Write-Host ""
    Write-Host "以下论文没跑成功: $($Failed -join ', ')" -ForegroundColor Red
}
