## JSON 数据模型

Pipeline 7.0 只保留一条领域真源链：冻结输入 → 按 Step 产生的 `formalization.json` Artifact 链，再从最终 formalization 派生最终索引与官方 Gaia IR。运行状态、工具记录和 Viewer 均为派生数据，不得保存第二份领域图。

```text
inputs/manifest.json
  → input.bundle.json
  → Step 1–4 formalization.json Artifact 链
      ├→ knowledge_index.json
      └→ gaia_ir.json
```

|文件|保存内容|定位|
|---|---|---|
|`run.json`|pipeline 配置、状态、attempt、checkpoint、输入 Artifact ID|运行状态真源|
|`inputs/manifest.json`|冻结输入的路径、kind、媒体类型和 SHA-256|输入身份真源|
|`artifacts/index.json`|Artifact ID、kind、路径、hash、生产 stage、metadata|Artifact 目录|
|`checkpoints/*.json`|成功 stage 的 Artifact ID 集合|恢复边界|
|`events.ndjson`|run/stage/finding/view 事件|运行记录，非领域真源|

### `input.bundle.json`

Importer 只消费 `source.paper_text` 与 `source.claims_final`，并生成一个输入包装。`sources` 中每项保留来源 Artifact ID、SHA-256、媒体类型和导入时实际读取的 `value`；当前流程不消费图片，因此 `figures` 固定为空数组。

```json
{
  "schema_version": "1.0.0",
  "sources": {
    "source.paper_text": [{"artifact_id": "…", "sha256": "…", "media_type": "text/markdown", "value": "…"}],
    "source.claims_final": [{"artifact_id": "…", "sha256": "…", "media_type": "application/json", "value": {}}]
  },
  "figures": []
}
```

### `formalization.json`

它是唯一领域真源。所有 Knowledge 节点保存在顶层 `knowledges`；策略、算子、Compose 和关系均以 Knowledge ID 引用节点。

```json
{
  "schema_version": "1.1.0",
  "revision": {"revision_id": "…", "supersedes": null, "parent_hash": null, "content_hash": "…"},
  "package": {"paper_id": "…", "namespace": "…", "name": "…", "version": "…"},
  "graph": {"nodes": [], "operators": [], "strategies": [], "composes": []},
  "knowledges": {},
  "workflow": {"source_records": [], "source_anchors": [], "non_reasoning_links": [], "weakpoints": [], "gaps": [], "revisions": []}
}
```

每个 Step 输出一个不可变的 formalization Artifact。`revision.supersedes` 与 `revision.parent_hash` 连接父版本，Artifact ID 与 SHA-256 标识实际文件；对象内部不再复制内容历史。`workflow.revisions` 仅为 schema 1.1.0 的兼容字段，始终写空数组。

### 派生产物与禁止项

| 文件                              | 规则                                                                                    |
| ------------------------------- | ------------------------------------------------------------------------------------- |
| `formalization_validation.json` | 每个 formalization Artifact 可有一份机械校验报告；非真源                                                  |
| `tool.*.response.json`          | 工具请求引用、raw、normalized、error、版本；运行记录，非领域真源                                          |
| `knowledge_index.json`          | 只在最终成功后生成一次，只包含在`graph`中引用过的节点（包括 `graph` 中直接引用的和作为 `operator` 的 `background` 等等间接引用） |
| `views/view_model.json`         | Viewer 从各 Step 的 formalization Artifact 派生图、搜索文档、图层和版本投影                                |
| `gaia_ir.json`                  | 唯一可消费的、官方 compiler 校验后的 Gaia IR                                                       |

### Viewer 版本接口

Step 1–4 的选择器以 formalization Artifact metadata 中的 `step` 为主键。Viewer 直接读取对应 Artifact 的完整内容，不读取或重建对象级 `history`；`overview / standard` 只改变展示粒度，不修改版本或领域数据。

---

## Step 1：导入命题集合
- `claims_final.json` 全量导入
	- `relations` 导入`workflow.non_reasoning_links`
	- 其余全部导入 `knowledges`，名称即为其 `id`
		- `is_pure_data=true` type为 `observation_claim`，作为一等实验观察节点进入 `graph.nodes`；`note` type为 `note` 不进入`graph`
		- 其他 `claim` 放入正式的 `graph.nodes` 中
- `paper_text.md`导入`input.bundle.sources`，按段落tag导入`workflow.source_anchors`
- relation 必须使用固定 expression 语法，方向只由 expression 决定，`connects` 仅校验同一组 claim。`X 是 Y 的例子或证据` 与 `X 推出 Y` 机械写为 `sources=X, target=Y`；Y 为合取组时按现有单 target link 契约展开为多条 link。等价、矛盾等对称关系按稳定 claim 顺序保存端点；未知语法、重复引用或 expression 与 `connects` 不一致时拒绝导入。

```json
{
  "graph": {
    "nodes": ["claim_12"],
    "operators": [],
    "strategies": [],
    "composes": []
  },
  "knowledges": {
    "claim_12": {
      "type": "claim",
      "content": {"canonical": "..."},
      "source_anchor_ids": ["anchor_claim_12"]
    }
  }
}
```

---

## Step 2：定位、补齐并写入实验 observation claim

### 2.1 机械化定位节点anchor
#### 定位已有节点
- 以 `knowledge` 中节点 `content` 的关键词检索 `paper`（含 Step 1 已全量导入的 `observation_claim`），排除仅有标题的段落；候选至少共享两个有效 token 且 token 余弦相似度不低于 0.2，才把最佳段落的 `workflow.source_anchors` 写入 `source_anchor_ids`。无可靠匹配时保持未定位，不以单个关键词强行锚定。
#### 重写已导入的 observation_claim
- Step 1 已把 `is_pure_data=true` 全量导入为 `observation_claim` 并进入 `graph.nodes`。Step 2 先对这批已有节点按 `S/A/M/R`（及可选 `B/U`）格式原文重写，保留原 ID、不另建并行候选节点；此步只重写 O，不生成 E、不建 relations。
#### 定位实验节点候选（补齐遗漏）
- 完成上一步后，再检索全文补齐 Step 1 遗漏的 observation_claim：在 `paper` 中检索 `image` 名称（例如 fig2），得到有引用图片的段落及其前后各一段，打包给 LLM 抽取实验 claim。

### 2.2 使用llm抽取实验claim
读取，将一个实验的信息组合成完整的 `S/A/M/R` 与可选 `B/U` 命题。
注意，在同一张图中，若可以读出独立的多个不同`S/A/B/M/R`，需要拆成多个 observation claim。
- S：实验设置／范围（Setting）
- A：处理、干预或实验对象（Action）
- B：可选的基线／对照组（Baseline）；论文没有明确提到时省略字段，不输出缺失占位符。
- M：测量指标（Measure）
- R：实验结果（Result）
- U：可选的、与当前 R 直接对应的不确定性结果（Uncertainty），例如具体 SD/SE、置信区间、数值范围或原文明示的跨运行变异。随机种子/replicate 数量、平均方式和“error bars represent mean ± standard deviation”属于实验或绘图协议；没有实际不确定性量值时不得作为 U。未报告 U 时省略字段，不写 `uncertainty not reported`。

#### 返回门禁
若发现已选中paper内容不足以提取实验claim，返回上一步机械门禁，扩大读取的paper范围

#### 写入实验claim
把所有实验claim写入 `formalization.json`的`knowledge`作为节点，并加入`graph`
- id: claim_O0*
- type: observation_claim
- content：由 LLM 在不改变 S/A/M/R 与可选 B/U 语义、范围、比较方向、条件、结果和不确定性的前提下，写成完整通顺的英文句子；不使用固定模板机械拼接，不增加、遗漏或改写事实。没有 B/U 时不输出对应字段或缺失占位符。
- source_anchor_ids: llm真正提取出该claim的段落tag（可以多个，含图片）

### 2.3 判重（已导入 vs 新补齐）
- 新抽取的 `observation_claim` 与已导入的 `observation_claim` 按内容 token 余弦相似度（阈值 0.85）判重；命中已导入的视为重复、丢弃新抽取的（已导入优先），未命中的赋新 id `claim_O0*`。
- 判重完成后，才统一对幸存的所有 observation_claim（含已导入重写与新增）生成等价 E。

### 2.4 使用 LLM 抽取 relations
#### 等价claim
判重后，对每个幸存的 observation_claim，编写一个等价的非实验表述的 claim E，满足
预测或现象 E ≡ 实际观测 O ，一一对应
- `E`：可由假说推出、可参与逻辑结构的现象 claim。
- `O`：具体实验或实际观察到的事实。
写入 `formalization.json` 的 `knowledges` 并加入 `graph.nodes`，中间用 `equivalence` operator 连接。作者态 operator 不保存 `conclusion`；官方编译投影在编译前机械生成 helper conclusion，且不回写作者态。

```json
{
  "knowledges": {
    "claim_E01": {
      "type": "claim",
      "content": {"canonical": "..."},
      "source_anchor_ids": ["anchor_paragraph_3"]
    }
  },
  "graph": {
    "nodes": ["claim_E01", "claim_O01"],
    "operators": [{
      "id": "operator_equivalence_E01_O01",
      "type": "equivalence",
      "variables": ["claim_E01", "claim_O01"]
    }]
  }
}
```
#### 其他relations
保持前面的输入。以每个 `E` 为中心，从当前实验候选 window 中检索 claim target：精确共享原文段落
`source_anchor`，或其段落 anchor 与当前候选段落距离不超过 2。实验内容不足而扩大读取 window 时，
按扩大后的 window 重新计算 relation 候选。note 不作为 relation source 或 target，只能在后续 Strategy/Operator
的 background 或解释字段中使用。再为候选的一般性 target 构造一个或多个证据集合，而非逐对判断。

每个候选关系由：
- `sources`：一个或多个 claim ID；
- `target`：恰好一个 claim ID；
- `expression`：只使用来源 ID、target ID 与固定逻辑词。

`expression` 候选：

1. 逻辑关系
   - `([1] 且 [2]) 推出 [3]`
   - `[1] 或 [2]`
   - `非 [1]`
   - `[1] 等价 [2]`
   - `[1] 与 [2] 矛盾`

2. 例子或证据关系
   - `[m] 是 [n] 的例子或证据`
   - `([m1] 和 [m2]) 是 [n] 的例子或证据`

只有当多条来源在同一局部语境中共同支撑同一 target，且删除任一来源会
明显改变关系含义时，才输出多来源 relation；否则保持单来源 relation。

若存在关系，写入 `workflow.non_reasoning_links`：
```json
{
  "id": "relation_image_03_mask_permutation_01",
  "sources": ["claim_m1", "claim_m2"],
  "target": "claim_n",
  "link_type": "imported_relation",
  "reasoning": false,
  "metadata": { "relation": {
    "expression": "([claim_m1] 和 [claim_m2]) 是 [claim_n] 的例子或证据",
    "relation_context_id": "image_03_mask_permutation"
  } }
}
```

`relation_context_id` 是 Step 2 实验候选级的审计边界：同一次图表/实验上下文产生的所有 target-centered relations 共享该 ID。它不表示这些 relation 必须合并，也不进入 Gaia 语义图。

`E`支持一般性结论时，语义候选默认为 `evidence`，但持久化仍固定使用
`link_type: "imported_relation"` 与 `reasoning: false`；不得写成 `[实验] 推出 [结论]`。

只有 target 是内容超出 E 的解释性假说、机制或一般规律时才输出 evidence；E 的改写、窄化实例、实验摘要或无关 claim 输出 none。`contradiction` 仅在 relation 已使用固定的“矛盾 / 不可同时成立”表达式时由 Step 3 机械解析；Pipeline 不另设语义判真门禁。

---

## Step 3：识别 weakpoint
### 3.1 固定关系与 bounded argument cluster
固定逻辑关系按下表机械归类；其余 evidence/推出 relation 先组成 bounded argument cluster，再联合规范化。完成后删除 `non_reasoning_links`。
#### 固定逻辑词

| expression 成分 | 确定性解析结果    | Gaia 层                   |
| ------------- | ---------- | ------------------------ |
| `且 / 和 / 与`   | `AND` AST  | `conjunction` Operator   |
| `或`           | `OR` AST   | `disjunction` Operator   |
| `非`           | `NOT` AST  | `negation` Operator      |
| `等价`          | 双方同真值      | `equivalence` Operator   |
| `矛盾 / 不可同时成立` | 固定表达式的二元不可同时为真结构 | `contradiction` Operator |
| `推出`          | 前件到后件的论证方向 | 标记 weakpoint             |
分别写入`graph.operators`


每个实验级 `relation_context_id` 形成一个 cluster 种子，包含该实验的全部 E 及其 target-centered candidate relations；随后只纳入一层与种子节点相邻、且共享原文 anchor 的非 E 粗关系。扩展时使用冻结的种子节点，不递归扩张。无 context 的旧 relation 或无法唯一归属的相邻 relation 各自形成独立 cluster。

`relation_context_id` 只限制联合审查范围，不是合并条件。Step 3 冻结一次 Step 2 Artifact 和 cluster 顺序；每个 cluster 基于同一冻结输入独立调用一次 LLM，各调用并行且只读。Pipeline 按冻结的 cluster 顺序确定性汇总响应，执行一次全局 relation 覆盖校验，最后只写入一个 Step 3 formalization Artifact，不在处理单个 weakpoint 后写回。LLM 可在现有节点范围内合并 relation、纠正方向、恢复相邻层级或拒绝错误候选，但不得创建语义命题。机械解析嵌套固定表达式时，Pipeline 可生成确定性的 AST helper claim 作为 Operator 中间结果；它只服务于形式结构，按 `formal_internal` 投影，Viewer 不把它当作普通公开 claim 展示。每条 candidate relation 必须在工具审计输出中恰好一次进入某个 weakpoint 的 `member_relation_ids`，或进入 `rejected_relation_ids`；这两个字段不写入领域 formalization。

若一个 self-contained 复合现象 claim B 严格包含多个原子现象 E，可输出一个多目标 deduction weakpoint `B → [E₁,E₂]`。若这些 E 同时被分别连向同一解释性假说 A，且 B 已忠实包含相关现象，优先输出相邻、非重复的 `B → A` abduction weakpoint，不再保留重复的 `E₁→A`、`E₂→A`。存在 `A→B→C` 时，除非原文独立明确陈述 `A→C`，不得保留跨层捷径。

规范化后写入 weakpoint：

```
{
  "id": "weakpoint_7",
  "payload": {
    "evidence_claim_ids": ["claim_3", "claim_4"],
    "target_claim_id": ["claim_9", "claim_10"],
    "reasoning_type": null,
    "evidence_anchor_ids": ["anchor_paragraph_p0012"],
    "expression": "原relation expression的内容"
  },
}
```

### 3.2 联合确定方向与 weakpoint 类别
LLM 在同一次 cluster 规范化中同时确定 `evidence_claim_ids`、一个或多个 `target_claim_id`、`expression` 和 `reasoning_type`。端点只能取自 cluster 的既有节点；Step 3 不补中间命题，中间命题由 Step 4 从原文提取并清洗。

不得仅因 evidence ID 以 `claim_E` 开头就机械判为 abduction；E 也必须满足“具体待解释现象”，target 也必须满足“解释性假说、机制或更一般规律”。一般命题到具体举例若缺少明确规则、范围与变量绑定，应判为 null，而非 deduction。

#### 分类定义

- `deduction`：证据/前提经一般规则、定义、模型或计算可推出 target；当前之所以是
  weakpoint，是因为规则、适用条件、变量绑定或中间步骤尚未显式化。
  - 典型形式：`P + rule + condition → C`
  - 不得把“实验观察支持一般结论”判为 deduction。

- `abduction`：由具体待解释现象 claim B 提出能够解释它们的假说、机制或一般规律；
  target 的内容超出 B 本身，且存在可竞争的替代解释。Step 2 生成的 E 是实验现象 B 的特例。
  - 典型形式：`B₁, B₂ → 最佳解释 H`
  - 多个实例视为同一假说的多条现象支持；当 B 是 E 时，O 只通过固定的 `E ≡ O` 接入。

- `analogy`：将源域中已建立的规律、机制或结构迁移到目标域；推理成立依赖于未展开的
  结构对应、变量映射或适用条件。
  - 典型形式：`G_src + mapping/bridge + S_target → C_target`
  - Step 4 目标：显式化类比桥梁 `M`、目标域条件 `S_target`，随后使用 entailment。
  - 仅有“相似”措辞而没有可识别的源域—目标域迁移时，不判为 analogy。

#### LLM 判定约束

输入仅包括 cluster 内的既有 self-contained claims、candidate relations、实验级 context 和可定位的原文片段。新 Artifact 的 `target_claim_id` 使用非空、去重的 claim ID 数组；旧 Artifact 的单个字符串按单元素数组读取且不改写原文件。

LLM 必须：

- 仅依据上述输入判定，不补造作者意图、隐藏前提、替代解释或跨 cluster 知识；
- 每条 candidate relation 必须被消费或拒绝恰好一次；multi-target 必须共享 premise、scope 和 reasoning family；
- 把“可由严格规则推出”与“观察后提出解释”严格区分；
- 当无法由输入识别分类时返回 `null`；
- 输出只允许为 `abduction`、`analogy`、`deduction` 或 `null`。

---

## Step 4：展开为细命题网络
### llm共用指令
- 依据weakpoint给定的 claim、expression 和原文，按以下模式展开 weakpoint
- 优先复用已有 claim，可补必要中间命题，但不得编造事实。
- Step 4 只输出命名 Strategy，写入 `graph.strategies`；不创建 Operator 或 `formal_expr`，不修改 Step 3 已有 Operator。Viewer 对 deduction、abduction、analogy 调用当前 Gaia `formalize_named_strategy()` 派生算符图，不显示这些 Strategy 节点；infer 按 Gaia 规则保留为可见 Strategy。派生算符不回写领域真源。
- 优先用已有的note节点作为strategy的background。如需要补充固定条件，把条件创造为knowledges中的note节点，不进入graph.nodes；在strategy.background中引用。具有独立不确定性的桥梁或前提必须是claim，不能藏入background。
- 作者态 Operator 如引用 background note，该引用只用于 strategy/operator 附带解释，不作为 Operator 的逻辑输入；Step 5 将它机械写入官方 `Operator.metadata.background`，并把相关 note 一并纳入 Gaia 图。
- 所有新建命题都需要定位原文，通过当前Pipeline私有副本
`agent-pipeline-v2/agent_pipeline_v2/claim_cleaner/run_single_text_pipeline_sync.py`。
该副本源自原清洗器；原目录不修改。零claim输入是合法的：写出既有的空分析结构；单条note输入保留为既有`claims_final.note`结构。
- 当前Step 4展开和命题清洗临时统一使用`deepseek-v4-flash`；清洗仍执行原14步流程及原机械校验，仅在清洗子进程期间通过已有`DEEPSEEK_MODEL`覆盖模型，结束后恢复环境。不需要Anthropic密钥或SDK，不跳过清洗。
- 输出：`{knowledges: {...}, strategies: [...]}`。每条strategy使用官方字段 `scope: "local"`、`type`、`premises`、`conclusion`、`background`；绑定Knowledge ID后，复用官方模型生成`strategy_id`。不输出概率、额外metadata或自定义算子。
- 运行环境须使用官方 `gaia-lang==0.5.0a5`；Step 5 compiler 在运行时执行精确版本门禁，版本不一致即失败。使用该版本官方 formalizer 与 `validate_local_graph`，不维护本地 Gaia 补丁。旧 Artifact 缺少 strategies 时按空集合读取且不改写原文件；新 Artifact 保存 strategies 容器。
- 原文不足以支撑所需推理结构时，返回空knowledges和strategies，保留weakpoint。以下公式描述推理语义，不要求Step 4持久化相应Operator。
- 一个 weakpoint 可以有多个 target。Gaia 官方 Strategy 仍保持单个 `conclusion`；Step 4 必须覆盖每个 target，为每个 target 输出独立的终端 Strategy，并可复用有原文依据的中间 Strategy。
- 若现有 Knowledge 不足以形成严格推理，Step 4 必须检查原文片段，提取缺失的中间命题、规则、变量绑定或条件，经命题清洗后接入子网络；不得用跨层直连掩盖缺口。原文仍不足时保留 weakpoint。

### deduction
M = A₁ ∧ A₂ ∧ ... ∧ Aₖ
M → C

显式化推导所需的规则、适用条件和变量绑定，纳入前提 Aᵢ。
保存 `type: "deduction", premises: [A₁, ..., Aₖ], conclusion: C`；已有M可直接使用`premises: [M]`，无需为合取序列化另造M。必要且有原文依据的中间命题可以由额外deduction策略建立。
规则与固定条件写入background note；严格推导的不确定性来自各命题前提是否成立。
不得将实验观察对一般结论的支持写成严格蕴含。

### abduction
A：假说。
AltExp：除 A 之外也能导致该现象的替代解释。已有原文支持的替代解释 claim 时可显式使用；没有合适 claim 时不在作者态创建占位 Knowledge，由官方 Gaia formalizer 在编译和 Viewer 投影时派生有内容的 alternative-explanation interface claim，Viewer 将其显示为 `AltExp` 占位符。
B：具体待解释的 self-contained 现象 claim，可以是非 E claim。
E：Step 2 生成的实验现象 B 特例；实际观测 O 已由固定 `E ≡ O` operator 连接。

(A ∨ AltExp) ≡ B

当 B 为 E 时：`(A ∨ AltExp) ≡ E ≡ O`。

若假说还需要条件才能产生现象，先显式化条件。
每个 B 保存一条 `type: "abduction", premises: [B], conclusion: A`；已有原文支持的替代解释时保存 `premises: [B, AltExp]`，顺序不可互换。固定适用条件写入 background。官方展开为 `disjunction.variables=[A, AltExp]`，再把析取结果与 B 连接为 equivalence。多个实例重复展开，共享解释假说 A；不得画成或解释成 `B→A`。当 B 是 E 时沿既有 equivalence 接入 O；普通 B 不虚构 O 节点。官方派生的 AltExp 不回写作者态 formalization。
Step 4 对每个已分类 weakpoint（包括以 E 为现象的 abduction）都调用 LLM，以允许从原文补入必要条件和中间命题。各 weakpoint 基于冻结的 Step 3 节点并行展开，响应按原 weakpoint 顺序确定性合并；命题清洗因使用共享工作目录和进程级模型覆盖，在临界区串行执行。


### analogy

G_src：源域中已经建立的规律、机制或约束。
M：类比桥梁，声明源域与目标域在相关变量、约束或因果结构上可对应。
S_target：目标域的具体条件、边界条件或适用范围。
V_target：迁移到目标域后的结论。

(G_src ∧ M ∧ S_target) → V_target

显式化变量映射及其保持的关系，不能只写“两者相似”。
不确定性在 M 是否成立；桥梁与条件给定后的推导必须严格。
保存 `type: "analogy", premises: [G_src, M], conclusion: V_target`，S_target作为明确的background note。多个源命题确需合并时，以有原文依据的deduction策略建立源域前提，不直接创建合取Operator。

### null
使用infer strategy：对 `target_claim_id` 中每个 target，机械映射一条 `type: "infer", premises: evidence_claim_ids, conclusion: target, background: []`，写入`graph.strategies`后移除对应weakpoint；原 Step 3 Artifact 保留来源。
不调用LLM，不增加命题，不将其视为严格deduction，也不编造条件概率。官方Strategy允许`conditional_probabilities`暂缺；需要概率推断时另行处理参数。

## Step 5：编码并校验 Gaia-IR

Step 5 只消费最终 Step 4 `formalization` Artifact，调用共享
`pipeline_harness.domain.compiler:Step5CompileGaiaIRPlugin`，并固定使用
`pipeline_harness.domain.compiler:Gaia05OfficialCompilerTool`。运行时必须精确匹配
`gaia-lang==0.5.0a5`；版本不一致、官方模型解析失败或
`validate_local_graph` 不通过时，Step 5 整体失败，不发布最终索引。

机械编译规则：

- 作者态 Knowledge ID 转为当前 package 下的官方 Gaia QID；`observation_claim` 在 Gaia IR 中按 `claim` 编码。
- 作者态固定 Operator 机械降低为官方 Operator；缺少显式 conclusion 时生成有内容的 `formal_internal` helper。Operator background note 随图编译，并写入 `Operator.metadata.background`，不成为逻辑 variable。
- `infer` Strategy 直接使用官方 Strategy；`deduction`、`abduction`、`analogy` 调用官方 formalizer。没有显式替代解释的 abduction 由官方生成有内容的 alternative-explanation interface claim，不回写作者态 formalization。
- 编译结果再次通过官方 `LocalCanonicalGraph` 解析和 `validate_local_graph` 校验，规范化后写出唯一 `gaia.ir` Artifact。

最终成功后，从同一 run 的 Step 1–4 immutable formalization Artifact 链派生一次
`knowledge.index`：只收录最终图直接或间接引用的 Knowledge；`first_seen_step`
取该 ID 首次出现的 Artifact step，`current_step=5`；incoming/outgoing reasoning
同时统计 Operator、Strategy 及其 background 引用；AST helper 标为
`formal_internal`。索引和 Gaia IR 都是 Step 4 领域真源的派生产物，互不作为对方输入。

完整装配的规范配置为 `agent-pipeline-v2.1/pipeline.step1-5.json`，
`pipeline_id=agent-pipeline-v2.1-step1-5`，版本统一为 `2.1.0`。配置不内置论文 ID、
package 名称或其他样例数据；这些值由输入或运行选项提供。

---

## 附录
### 目前viewer
file:///Users/catherinewu/Documents/gaia-ir/01_paper_sample_one_ticket/gold_standard_new/generated/runs/run_20260827T024814Z_9023a33f/views/viewer.html?dev=1

### 清洗工具
当前Pipeline使用：`agent-pipeline-v2/agent_pipeline_v2/claim_cleaner/run_single_text_pipeline_sync.py`  
（原始来源保留在`paper_graph2ir/`，不由Pipeline修改。）

- 功能：输入一段独立文本，抽取并清理为可独立核验的命题；会补足指代/条件、必要时关联上下文，输出最终命题网络。
- 输入：`text`（命令行文本）或 `--file`（文本文件）；默认需 `ANTHROPIC_API_KEY`，显式设置 `DEEPSEEK_MODEL` 时改用 `DEEPSEEK_API_KEY`（Step 4 当前使用此覆盖）。
- 输出：`data/<自动生成paper_id>/claims_final.json`

核心输出字段：

- `claim[]`：`conclusion`, `text`（清理后的完整命题）, `is_pure_data`, `needs_more_context`, `number`
- `note[]`：被命题引用、但不单独构成命题的补充文字
- `relation[]`：命题间的逻辑、例子或证据关系

### `relation.expression` 

1. 逻辑关系：由固定逻辑词组合
    - `且` / `和` / `与`：合取，例如 `([1] 且 [2]) 推出 [3]`
    - `或`：析取
    - `非`：否定
    - `推出`：蕴含 / 推理方向
    - `等价`：等价
    - `矛盾`：冲突、不可同时成立
    - `()`：分组
    - `[N]`：引用 claim 编号
2. 例子或证据关系
    - `[m] 是 [n] 的例子或证据`
    - `[m] 是 ([n1] 和 [n2]) 的例子或证据`

### 官方 Gaia 图关系

| 层                  | 官方类型                                                                                                                                                                    |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 确定性结构约束 `Operator` | `implication`、`negation`、`conjunction`、`disjunction`、`equivalence`、`contradiction`、`complement`                                                                         |
| 推理超边 `Strategy`    | `infer`、`associate`、`deduction`、`reductio`、`elimination`、`mathematical_induction`、`case_analysis`、`abduction`、`analogy`、`extrapolation`、`support`、`compare`、`induction` |

其中：

- `equivalence`（↔）、`contradiction`（⊗）、`complement`（⊕）是**结构关系**，不是 weakpoint；
- “实验/例子/证据支持结论”通常应先成为 `Strategy` 候选，如 `abduction`、`induction`、`support`；
- `weakpoint` 只是“这个 Strategy 尚未展开、存在桥梁或替代解释”的审查状态；
- `not_reasoning` 则不进入 Gaia graph；
- `source`、`split_from`、导入 relation 等属于 Pipeline 的溯源/工作流链接，不是 Gaia IR 的推理或逻辑 primitive。