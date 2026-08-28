# test_new_labels2.ps1
# 去掉了proof规则里"assertion必须已经在前面完整说过"的时序限制,其它不变
# 只重跑 paper2_conclusion_6 和 paper2_conclusion_9 这两条

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$promptTemplate = @'
You will receive a piece of "claim" text (content) from a scientific paper. Please split this text into several consecutive segments (concatenated in order, they must match the original text exactly - no omissions, no duplications, no rewording), and label each segment with one of the following seven categories:

- "assertion [n]": an independently valid scientific claim/finding stated in the text; merely describing what this paper does or what method is used does not by itself count as an assertion. A piece of content may contain multiple assertions, and it may also contain no assertion at all. Note: [n] is the assertion's 1-indexed number - regardless of how many assertions this content contains, every assertion's label must include its own number. [n] must be a plain Arabic numeral (e.g. 1, 2, 3) - not a letter, word, or Roman numeral. A semantically complete assertion is not necessarily contiguous in the original text - other elaboration, examples, or proofs may be inserted in the middle of it. If an assertion is made up of several separate spans in the original text, label each of these spans with the same assertion number; segments sharing the same assertion number, when concatenated in their original order, must together form a semantically complete sentence. If splitting out some piece of content to label it as a different category would break the semantic completeness of an assertion segment, do not split it out from that assertion segment. Content inside parentheses is often content of another category that interrupts an assertion; however, content that interrupts an assertion is not necessarily enclosed in parentheses - it can occur without parentheses as well. If a single piece of content contains more than one assertion, they must be labeled as separate segments - they cannot be merged into one "assertion" segment.

- motivation: states the purpose of some assertions.

- elaboration: gives a definition or additional explanation for some term, method, setting, or other detail.

- "example of the assertion [n]" or "example of the assertions [n_1,n_2,...]": corroborates or illustrates an assertion [n] or multiple assertions [n_1, n_2, ...] that has already been fully and independently stated earlier. Note: this label may only be used for examples of assertions, not for examples of any other category (such as motivation or elaboration) - if an example is illustrating a text of another catogory, it should be merged into that same segment and given that same label, rather than being split out and labeled "example of the assertion".

- "proof of the assertion [n]" or "proof of the assertions [n_1,n_2,...]": provides evidence or argumentation supporting an assertion [n] or multiple assertions [n_1, n_2, ...] - the assertion(s) being supported do not need to appear before this segment; they may come before or after it. Note: this label may only be used for evidence/argumentation supporting assertions, not for evidence/argumentation supporting any other category (such as motivation or elaboration) - if it is supporting a text of another catogory, it should be merged into that same segment and given that same label, rather than being split out and labeled "proof of the assertion".

- "relation of the assertions [n_1,n_2,...]": a short logical connector expressing the logical relationship between two or more assertions (e.g. implication, contradiction, or a therefore-style conclusion), referencing the numbers [n_1, n_2, ...] of the assertions it connects. A relation segment is usually just one or a few words, including but not limited to phrases like "This implies", "This contradicts with", "Therefore". Note: only a logical relationship between at least two distinct, separately-numbered assertions counts as a relation - a logical connector occurring entirely within a single assertion does not count, nor does a relationship between an assertion and a segment of any other category. Only a genuinely logical relationship counts; non-logical relationships do not qualify.

- other: content that doesn't belong to any of the above six categories.

Output strictly in the following JSON format, with no other text:
{
  "segments": [
    {"text": "......", "label": "assertion [n]"},
    {"text": "......", "label": "motivation"}
  ]
}

Now process the following content:

'@

$testCases = @(
    @{
        Name    = "paper2_conclusion_6 (EDGE-POPUP)"
        Content = @'
EDGE-POPUP, an approach that trains per-parameter scores (targeted to find strong tickets, i.e., subnetworks that perform well at initialization without further training), recovered a strong ticket at sparsity $\rho\approx 0.5$ for the Circle task in its original form but failed to retrieve planted tickets at much lower sparsities. Extending EDGE-POPUP with an annealing schedule that gradually reduces the target sparsity across score-training rounds (specifically, reducing sparsity to $\rho^{i/10}$ in round $i$) improved its performance: the modified method discovered stronger (sparser) strong tickets (e.g., achieving reasonable accuracy near $\rho=0.1$ in some settings), representing an order-of-magnitude sparsity improvement over the original behavior; nevertheless, in experiments that planted a trained weak ticket (a subnetwork that requires training) of sparsity $\rho=0.01$ into VGG16 on CIFAR10, EDGE-POPUP could not recover the planted baseline sparse strong ticket, indicating the method still falls short of planted ground-truth sparsity in realistic CNN settings.
'@
    },
    @{
        Name    = "paper2_conclusion_9 (planting framework / Because...indicates that)"
        Content = @'
Using the planting framework, in an experiment where a trained weak ticket (a subnetwork that requires further training) of sparsity $\rho=0.01$ (discovered by the SYNFLOW pruning method and trained on CIFAR10) was planted back into a VGG16 neural network and then given to the EDGE-POPUP pruning method, EDGE-POPUP failed to retrieve the planted $\rho=0.01$ strong ticket (a subnetwork that performs well at initialization) in this realistic convolutional neural network setting. Because the planted ticket is known to exist inside the host by construction, this failure indicates that the inability of pruning methods (here EDGE-POPUP) to find extremely sparse strong tickets in practice is due to algorithmic limitations of current methods rather than fundamental nonexistence of such tickets in randomly initialized networks.
'@
    }
)

$headers = @{
    "x-api-key"         = $env:ANTHROPIC_API_KEY
    "anthropic-version" = "2023-06-01"
    "content-type"      = "application/json"
}

foreach ($case in $testCases) {
    Write-Host ""
    Write-Host "===================================================="
    Write-Host "处理: $($case.Name)"
    Write-Host "===================================================="

    $fullPrompt = $promptTemplate + $case.Content

    $body = @{
        model      = "claude-sonnet-5"
        max_tokens = 5000
        messages   = @(
            @{
                role    = "user"
                content = $fullPrompt
            }
        )
    } | ConvertTo-Json -Depth 10

    try {
        $response = Invoke-RestMethod -Uri "https://api.anthropic.com/v1/messages" `
            -Method Post -Headers $headers -Body $body -ErrorAction Stop

        $textBlocks = $response.content | Where-Object { $_.type -eq "text" } | ForEach-Object { $_.text }
        $rawAnswer = $textBlocks -join ""

        Write-Host "stop_reason: $($response.stop_reason)"
        Write-Host "content块类型: $($response.content | ForEach-Object { $_.type } | Out-String)"
        Write-Host $rawAnswer
    }
    catch {
        Write-Host "调用出错: $($_.Exception.Message)" -ForegroundColor Red
        if ($_.ErrorDetails) {
            Write-Host $_.ErrorDetails.Message -ForegroundColor Red
        }
    }
}
