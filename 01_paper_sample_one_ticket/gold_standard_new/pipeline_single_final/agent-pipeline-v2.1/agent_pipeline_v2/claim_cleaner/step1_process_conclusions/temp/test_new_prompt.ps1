# test_new_prompt.ps1
#
# 直接用PowerShell调Anthropic API，测试新版 step1a_prompt_template_v2.txt。
# 不经过 claude_api_call.py，独立跑。
#
# 用流式(streaming, stream=true)方式调用，而不是一次性等完整响应：
# 之前用同步方式，xhigh档位思考时间长，连接长时间没有数据往回传，容易被
# 网络设备当成"空闲连接"中途掐断（Anthropic官方文档也提到过这个风险）。
# 流式方式下，模型每生成一点内容就有数据陆续传回来，连接一直在传输，
# 能大幅降低被中途掐断的概率。
#
# 用HttpClient而不是Invoke-RestMethod：Windows PowerShell 5.1的
# Invoke-RestMethod/Invoke-WebRequest对请求体、响应体的编码检测不稳定，
# 容易把中文/特殊符号读错、发错（表现为"论据"变成"è®ºæ®"这种乱码）。
# HttpClient请求和响应两头都显式声明UTF-8，不依赖PowerShell自己去猜编码。
#
# 用法：
#   1. 确保 $env:ANTHROPIC_API_KEY 已经设置好
#   2. 确保 step1a_prompt_template_v2.txt 跟这个脚本放在同一个目录下
#   3. cd 到这个目录，然后:  .\test_new_prompt.ps1

$ErrorActionPreference = "Stop"

if (-not $env:ANTHROPIC_API_KEY) {
    Write-Host "没检测到 ANTHROPIC_API_KEY 环境变量，先设置好再跑。" -ForegroundColor Red
    exit 1
}

Add-Type -AssemblyName System.Net.Http

$promptTemplate = Get-Content -Path ".\step1a_prompt_template_v2.txt" -Raw -Encoding UTF8

# ---------------------------------------------------------------------------
# 本轮只重跑之前撞到max_tokens=20000上限的这1条，这次用max_tokens=40000
# ---------------------------------------------------------------------------
$testContents = [ordered]@{

    "conclusion_Trojan检测_Therefore式2元relation_公式密集" = @'
In the specific iterative magnitude pruning (IMP) regime adopted for Trojan analysis — where after each pruning round the surviving weights are not rewound to their initialization and all finetuning uses only the potentially poisoned dataset $\mathcal{D}_\mathrm{p}$ (no clean data) — the Trojan score is defined to quantify the pruning-induced stability of Trojan tickets by measuring the linear-mode connectivity (LMC) error barrier between the un-finetuned Trojan ticket $(m\odot\theta)$ and the $k$-step finetuned Trojan ticket $(m\odot\theta^{(k)})$. Here, $\theta$ are the parameters of the dense model trained on $\mathcal{D}_\mathrm{p}$, $m\in\{0,1\}^d$ is the binary pruning mask, $\odot$ denotes elementwise product, and $\theta^{(k)}$ denotes $\theta$ after $k$ gradient steps of finetuning on $\mathcal{D}_\mathrm{p}$. Given a model training error functional $\mathcal{E}(\cdot)$ evaluated on $\mathcal{D}_\mathrm{p}$, the LMC error barrier is $$e_{\sup}(\phi_1,\phi_2)=\max_{\alpha\in[0,1]}\mathcal{E}(\alpha\phi_1+(1-\alpha)\phi_2).$$ Setting $\phi_1 = m\odot\theta$ and $\phi_2 = m\odot\theta^{(k)}$, the Trojan score is $$\mathcal{S}_{\mathrm{Trojan}} \;=\; e_{\sup}(m\odot\theta,\;m\odot\theta^{(k)}) \;-\; \frac{\mathcal{E}(m\odot\theta)+\mathcal{E}(m\odot\theta^{(k)})}{2}.$$ Under the described IMP regime, the Trojan score is approximately zero for non-Trojan (clean) models because the linear interpolation error barrier equals the average of the endpoints' errors. A significantly positive peak in $\mathcal{S}_{\mathrm{Trojan}}$ along the pruning path indicates an atypical pruning stability fingerprint associated with a backdoor. Therefore, computing $\mathcal{S}_{\mathrm{Trojan}}$ along the IMP pruning path yields a data-free detector that identifies pruning levels where Trojan information is isolated.
'@
}

$httpClient = New-Object System.Net.Http.HttpClient
$httpClient.Timeout = [System.TimeSpan]::FromMinutes(10)
$httpClient.DefaultRequestHeaders.Add("x-api-key", $env:ANTHROPIC_API_KEY)
$httpClient.DefaultRequestHeaders.Add("anthropic-version", "2023-06-01")

foreach ($key in $testContents.Keys) {
    Write-Host "===================================================================" -ForegroundColor Cyan
    Write-Host "测试: $key" -ForegroundColor Cyan
    Write-Host "===================================================================" -ForegroundColor Cyan

    $fullPrompt = $promptTemplate + $testContents[$key]

    $bodyObj = @{
        model         = "claude-sonnet-5"
        max_tokens    = 40000
        thinking      = @{ type = "adaptive" }
        output_config = @{ effort = "xhigh" }
        stream        = $true
        messages      = @(
            @{ role = "user"; content = $fullPrompt }
        )
    }
    $bodyJson = $bodyObj | ConvertTo-Json -Depth 10

    # 请求体：显式用UTF8编码构造StringContent
    $httpContent = New-Object System.Net.Http.StringContent($bodyJson, [System.Text.Encoding]::UTF8, "application/json")

    $outputText = ""
    $usage = $null
    $stopReason = $null
    $failed = $false

    try {
        # ResponseHeadersRead：拿到响应头就往下走，不等整个响应体传完，
        # 这样才能一边收流一边处理，而不是被HttpClient自己先攒完整个响应
        # 注意：HttpCompletionOption 只能配合 SendAsync 用，PostAsync 没有
        # 接受这个参数的重载，所以这里手动构造 HttpRequestMessage 再发送。
        $request = New-Object System.Net.Http.HttpRequestMessage(
            [System.Net.Http.HttpMethod]::Post,
            "https://api.anthropic.com/v1/messages"
        )
        $request.Content = $httpContent

        $httpResponse = $httpClient.SendAsync(
            $request,
            [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead
        ).Result

        if ($null -eq $httpResponse) {
            throw "PostAsync 返回了空值（大概率是连接中途断开）"
        }

        if (-not $httpResponse.IsSuccessStatusCode) {
            $errBytes = $httpResponse.Content.ReadAsByteArrayAsync().Result
            $errText = [System.Text.Encoding]::UTF8.GetString($errBytes)
            Write-Host "HTTP错误 $([int]$httpResponse.StatusCode):" -ForegroundColor Red
            Write-Host $errText -ForegroundColor Red
            Write-Host "如果错误提示是关于 effort 档位不支持，把脚本里两处 xhigh 改成 high 再试一次。" -ForegroundColor Yellow
            $failed = $true
        }
        else {
            $stream = $httpResponse.Content.ReadAsStreamAsync().Result
            $reader = New-Object System.IO.StreamReader($stream, [System.Text.Encoding]::UTF8)

            while (-not $reader.EndOfStream) {
                $line = $reader.ReadLine()
                if ([string]::IsNullOrEmpty($line)) { continue }
                if (-not $line.StartsWith("data: ")) { continue }

                $jsonPart = $line.Substring(6)
                $eventObj = $jsonPart | ConvertFrom-Json

                switch ($eventObj.type) {
                    "content_block_delta" {
                        if ($eventObj.delta.type -eq "text_delta") {
                            $outputText += $eventObj.delta.text
                        }
                    }
                    "message_start" {
                        $usage = $eventObj.message.usage
                    }
                    "message_delta" {
                        if ($eventObj.usage) { $usage = $eventObj.usage }
                        if ($eventObj.delta.stop_reason) { $stopReason = $eventObj.delta.stop_reason }
                    }
                }
            }
            $reader.Close()
            $stream.Close()
        }
    }
    catch {
        Write-Host "调用出错: $_" -ForegroundColor Red
        $failed = $true
    }

    if ($failed) {
        Write-Host ""
        continue
    }

    if ($usage) {
        Write-Host "用量: input=$($usage.input_tokens) tokens, output=$($usage.output_tokens) tokens" -ForegroundColor Magenta
    }
    if ($stopReason -eq "max_tokens") {
        Write-Host "警告：这次是被 max_tokens 截断的（stop_reason=max_tokens），下面的内容大概率不完整、解析不出来！" -ForegroundColor Red
    }
    elseif ($stopReason) {
        Write-Host "stop_reason: $stopReason" -ForegroundColor DarkGray
    }

    $outFile = ".\test_output_$($key -replace '[^\w]', '_').json"
    $outputText | Out-File -FilePath $outFile -Encoding utf8

    Write-Host $outputText
    Write-Host ""
    Write-Host "已保存到: $outFile" -ForegroundColor Green
    Write-Host ""
}

$httpClient.Dispose()
Write-Host "全部测试完成。"
