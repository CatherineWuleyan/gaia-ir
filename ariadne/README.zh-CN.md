# Ariadne

**问题驱动的领域推理图。**

[English](README.md) · [示例](#示例) · [快速开始](#快速开始) · [配置](#配置) · [命令](#命令)

![飞机仿真设计问题图](docs/images/aircraft-question-graph.png)

Ariadne 把领域问题变成一张可检查的图：论文、claim、候选答案、推理链都在图
上。它不直接给一段聊天式答案，而是给出：

- 从问题生成的**候选答案**（root，R1–R5）；
- 支持、限定或反驳每个答案的 **claim 与证据**；
- claim 之间的**类型化关系**，包括跨论文关系；
- 到达每个 root 的**全部多步推理链**（Markdown + Mermaid）。

名字取自 Ariadne 的线团：在迷宫里靠它找路。这里“线”就是答案层和通向答案的
推理路径。

```text
问题
  -> LKM 查询规划 + 多查询检索 + 多信源交叉验证 + 图探测选论文
  -> paper2ir      (论文图 -> claims_final)
  -> single-paper  (claims -> Gaia IR package)
  -> merge         (多篇 package -> 领域图)
  -> Step 6        (候选答案 root + 类型化链接 + 推理链)
  -> HTML 问题图 + 全部链条 Markdown
```

Ariadne 是维护中的 Gaia IR 三套 pipeline 之上的**薄产品层**：不复制、不改写
pipeline 内部实现，只负责编排、隔离每次运行，并补上问题驱动的答案层和交付物。

## 下载

不用 clone，直接下载自包含安装包：

[ariadne-0.5.0-bundle.tar.gz](https://github.com/CatherineWuleyan/gaia-ir/releases/download/v0.5.0/ariadne-0.5.0-bundle.tar.gz) ·
[ariadne-0.5.0-bundle.zip](https://github.com/CatherineWuleyan/gaia-ir/releases/download/v0.5.0/ariadne-0.5.0-bundle.zip) ·
[SHA256SUMS](https://github.com/CatherineWuleyan/gaia-ir/releases/download/v0.5.0/SHA256SUMS)

包内含 Ariadne、三套 pipeline、离线飞机示例和 `install.sh`。自己构建：

```bash
./packaging/build_release.sh
```

## 状态与边界

引用下面任何数字之前先读这一节。

- **图没有量化评估。** claim 抽取、关系类型、root 选择都没有 precision/recall
  或标注一致性数据。`benchmark/e0` 只有 2 道题的 checker pilot，physics 示例
  只有 1 张人工复核的图。本页的计数衡量的是规模，不是正确性。
- **typed support chain 很稀有。** 冻结飞机图是 0 条，8 篇实时运行也是 0 条：
  它导出的 17 条是阅读序列和 `infer` 路径。系统不会悄悄把任何边升级成
  deduction，所以真正需要多步演绎的问题目前会显示空链。
- **飞机图的跨论文改类型是 LLM 审计**，不是人工复核（`edge_roles.json` 记录了
  模型）。只有 physics 示例是人工复核。
- `demo` 在冻结飞机图上做过离线测试，会产出两个交付物。
- 一次 8 篇飞机自动运行已端到端完成：**184 条 claim、105 条关系、5 个候选
  答案、33 条 answer links、17 条导出路径**（5 条同论文序列 + 12 条图路径）。
  这次用 `--top-k 8`；k 是调用时参数，不是硬编码的论文数。
- 8 篇里有 3 篇的图是 `bohr lkm extract` 自动补出来的：Interactive Cockpit
  （62 节点 / 15 conclusions）、Cybermatrix（46 / 10）、Flying-V（56 / 12）。
  原生图和补建图走同一套 selection/materialise 流程。
- 宽问题上仍有很多高相关候选是空原生图。IEEE/AIAA 等闭源 PDF、以及超过 50 页
  的 PDF 会在 `selection.json` 里记录 `extract_skip_reason` 后跳过，不伪造图。
- LKM 补建图路径也在真实 arXiv 论文（Maestro）上跑通：87 节点 / 88 边 /
  19 conclusions。空原生图 → 补建图的判定有单元测试覆盖；只有存在 PDF/arXiv
  源时才会触发。
- `full` 在花钱前预检配置、Python、pipeline 文件和 Bohr 登录。建议先用
  `--top-k 2 --no-fulltext` 检查质量和成本。
- full 模式链条质量取决于 paper2ir、single、merge 和问题层。Ariadne 只报告这些
  阶段产生的东西，不伪造缺失证据，也不把 `infer` 升级成 `deduction`。
- viewer 默认显示全部图路径。能画出来不代表自动成为 support chain。
- Physics 示例是复核过的静态图，不是一次新的实时运行；它展示答案层和复核界面
  的形态。


## 示例

### 1. 怎么完成飞机的仿真设计？

> **问题：** 怎么完成飞机的仿真设计？
> **领域：** 概念设计、飞行动力学、气动建模、MDO、硬件在环验证。

![飞机问题图](docs/images/aircraft-question-graph.png)

冻结的飞机图包含 **13 篇论文、311 条 claim、223 条边、93 条 integration
delta、5 个候选答案**。

**它包含 0 条 typed support chain。** 只有 deduction 可入链，而这张图没有任何
deduction；导出器产出的是：

- **7 条 same-paper 序列**：同一篇论文的直接支持 claim，按 claim 编号排列。
  这是阅读顺序，不是证明；
- **100 条 exploratory path**：`infer` 关系，作为上下文展示。

| Root | 候选答案 | same-paper | exploratory | 判定 |
| --- | --- | ---: | ---: | --- |
| R1 | 集成概念设计环境 | 2 | 18 | multi-paper direct support（7 + 3 条） |
| R2 | 嵌套多学科优化 | 2 | 24 | multi-paper direct support（5 + 3 条） |
| R3 | 非线性飞行动力学建模 | 1 | 24 | single-paper core（7 + 1 条） |
| R4 | 飞行数据模型校正 | 1 | 20 | single-paper core（9 + 1 条） |
| R5 | 硬件在环验证 | 1 | 14 | single-paper core（8 + 1 条） |

判定的依据是**第二篇论文贡献了多少条 claim**。R3–R5 结构上确实有 2 篇论文，
但第二篇只提供 9 条里的 1 条，所以不构成跨论文互证；R1、R2 构成。

看图时重点看：

- 两个红色大 root（多论文）和三个橙色 root（单论文核心）——颜色现在跟着判定走；
- 绿色 **support** 链接；
- **same-paper** 是按论文和 claim 编号排列的阅读顺序，不是逻辑证明；
- 虚线 **analogy** / **infer** 是可见的上下文，但**不会**被升级成 deduction，
  也不构成 support chain。

交付物：

- [飞机问题图 viewer](../outputs/aircraft/final/domain_viewer_question.html)
- [指向 R1–R5 的全部链条](../outputs/aircraft/final/ALL_CHAINS_TO_ROOTS.md)
- [答案层 JSON](../outputs/aircraft/final/answer_layer.json)
- [冻结 demo 数据](../outputs/aircraft/final/)

离线运行（不需要 key，不调用 LKM）：

```bash
./ariadne/run.sh demo
```

### 2. 4H_a-NbSe2 的面内 B_c2 由哪条对破缺通道决定？

> **问题：** 在块体 4H_a-NbSe2（非中心对称多型）中，面内上临界场
> B_c2∥ 由哪条对破缺通道决定：塞曼（经 Ising SOC 抬高）还是轨道
>（含有限动量/FFLO）？

![Physics 问题图](docs/images/physics-question-graph.png)

复核版 Physics 图包含 **18 篇论文、500 条 claim、654 个节点、1298 条边、
14 个理论层节点**。这个例子说明答案层必须把两个“兼容但竞争”的解释分开：

- **R1**：Ising 保护压制塞曼通道，由轨道/层间通道决定主要量级；
- **R2**：若 γ ≈ 31/5.5 ≈ 5.6，纯轨道通道在数值上可以自足；
- 真正未决的是 Ising SOC 是否**必要**，而不是它是否存在。

Physics 例子也展示了产品的诚实边界：图里有 9 条矛盾边，包含跨对称性、
跨材料的证据。viewer 会显示它们，但不会悄悄改写成 support。

交付物：

- [Physics 复核版 viewer](../outputs/physics/final/domain_graph_full.reviewed.html)
- [答案层 JSON](../outputs/physics/final/answer_layer.reviewed.json)
- [Mentor 复核记录](../outputs/physics/final/MENTOR_REVIEW.md)
- [目标问题](../outputs/physics/build/target_question.json)

GitHub 会把 HTML 当源码显示。克隆后用浏览器打开，或在仓库根目录运行
`python3 -m http.server` 再访问。

## 快速开始

### 环境要求

| 依赖 | 用途 |
| --- | --- |
| Python **3.11+** | harness 和 pipeline 代码 |
| `gaia-lang==0.5.0a7` | single 阶段使用的 Gaia IR 编译器 |
| `bohr` CLI **2.2+** | 实时 LKM 检索、论文图、可选 PDF 解析 |
| DeepSeek 或 OpenAI-compatible API key | paper2ir、single、merge、Step 6 |
| Bohrium 账户 + access key | 实时 LKM 调用（`demo` 不需要） |

公开仓库已经包含 Ariadne 需要的 pipeline 代码：
`pipelines/clean_claims`、`pipelines/single_paper`、`pipelines/merge` 和
`weekly_reports/export_single_artifacts.py`。

### 1. Clone 并安装

```bash
git clone https://github.com/CatherineWuleyan/gaia-ir.git
cd gaia-ir

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --pre -r ariadne/requirements.txt
pip install -e agent
```

`run.sh` 按以下顺序选择解释器：`$ARIADNE_PYTHON`、`.venv-gaia-a7`、`.venv`、
`python3`。也可以把解释器写进配置（见下文）。

### 2. 安装 Bohr CLI

```bash
npm install -g bohr-cli
bohr version
```

如果本机已经登录 `bohr`，Ariadne 可以直接用当前 active profile。想给 Ariadne
单独一个 profile，用下一步的 `ariadne auth bohr`。

### 3. 配置 API key 和 Bohr 账户

交互式（推荐）：

```bash
./ariadne/run.sh config init
```

非交互式：

```bash
export DEEPSEEK_API_KEY="sk-..."
export BOHR_ACCESS_KEY="..."

./ariadne/run.sh config init --no-input \
  --api-key "$DEEPSEEK_API_KEY" \
  --bohr-access-key "$BOHR_ACCESS_KEY"
```

配置写到 `~/.config/ariadne/config.json`，权限 `0600`。Ariadne 不会打印 key；
`config show` 和 `config get` 都会打码。

然后登录 Bohr：

```bash
./ariadne/run.sh auth bohr
```

它创建/更新 Bohr profile `ariadne`，不改你原来的 default profile。想用浏览器
或 device flow：

```bash
./ariadne/run.sh auth bohr --device
```

最后检查：

```bash
./ariadne/run.sh doctor
```

`doctor` 会报告配置来源、模型 endpoint、Bohr 登录、pipeline 文件，以及
demo / 离线 full / 实时 LKM full 是否就绪，不会输出 secret。

### 4. 跑离线 demo

```bash
./ariadne/run.sh demo
```

在冻结的飞机图上只跑问题层：不调用 LKM，也不调用 LLM。输出在
`outputs/ariadne_runs/`：

```text
outputs/ariadne_runs/run_<timestamp>_<question>/
  domain_viewer_question.html   # 自包含的问题图
  ALL_CHAINS_TO_ROOTS.md        # 每个 root 的全部多步链
  run.json                      # 问题、阶段耗时、运行时摘要
  04_question/                  # answer_layer.json、view model、evaluation
```

打开 viewer：

```bash
open outputs/ariadne_runs/run_*/domain_viewer_question.html      # macOS
xdg-open outputs/ariadne_runs/run_*/domain_viewer_question.html  # Linux
```

### 5. 跑实时问题

先用**两篇论文**检查质量和成本：

```bash
./ariadne/run.sh ask "怎么完成飞机的仿真设计？" --top-k 2 --yes
```

现在 Ariadne 会自己选论文：LLM 查询规划器把问题扩成多条 LKM 查询，候选论文
经独立信源交叉验证后去重排序，再逐个探测论文图；空图、没有 conclusion 的图
和过薄的图会被跳过，然后才进入 paper2ir。完整审计写在
`01_retrieval/selection.json`。

不花钱先看计划：

```bash
./ariadne/run.sh ask "怎么完成飞机的仿真设计？" --top-k 2 --dry-run
```

说明：

- `--yes` 授权本次运行的计费 LKM 调用。不传时，交互终端会询问；批处理环境
  会直接失败，不会挂住。
- `--top-k 2` 控制 paper2ir 的成本和延迟。每篇论文 paper2ir 有 **14 个 LLM
  步骤**，之后 single 和 merge 还会调模型。
- `--no-fulltext` 跳过 arXiv PDF 下载/解析，改用图派生文本。
- `--no-query-planner` 只用确定性 fallback 查询。
- `--queries N` / `--candidate-limit N` 调整检索宽度和图探测数量。
- `--no-cross-check` 关闭多信源交叉验证；`--external-discovery` 允许「多个独立
  源一致、但 LKM 漏检」的论文进入候选池。
- `--papers papers.json` 用本地 spec 替代自动检索（见
  [离线论文 spec](#离线论文-spec)）。

## 自动检索

### 个人知识库与用户文档

指定 `--library PATH` 后，Ariadne 会先召回库内相关命题，补齐前提、条件和反证，
检查证据覆盖度。证据足够时只运行问题答案层；已有材料缺少关系时局部重整；
材料不足时围绕缺口补检索，只处理新增或内容变化的论文，再增量接入相关局部。
用户文档先导入，再检查是否需要补检索；`--papers` 仍保持显式指定的语料范围。
每次最多补检索一轮，仍未解决的缺口会显示在答案 caveat 和 `run.json` 中。
明确要求“最新”的问题会重新检索核验。

```bash
# 首次建立个人库；以后沿用同一路径，连续追问可复用已验证的包与关系
./ariadne/run.sh ask "怎么完成飞机的仿真设计？" --library outputs/my_library --top-k 2 --yes
./ariadne/run.sh ask "如何利用风洞数据校准模型？" --library outputs/my_library --yes

# 选择自己的文件；可重复指定 --documents，也可一次列出多个文件
./ariadne/run.sh ask "这些材料支持什么结论？" --library outputs/my_library \
  --documents /absolute/path/paper.pdf /absolute/path/notes.md --yes

# 只看计划；不写库、不上传、不调用模型
./ariadne/run.sh ask "这些材料支持什么结论？" --library outputs/my_library \
  --documents /absolute/path/notes.md --dry-run
```

导入依赖 `reportlab` 和 `pypdf`，已列入 `ariadne/requirements.txt`。PDF 必须未加密、
不超过 64 MiB／50 页。Markdown 使用保留原始文本与公式源码的 PDF 作为 LKM 传输
格式，原始 Markdown 继续用于单篇处理；不会读取文档中的远程链接或图片。
PDF 上传后走现有 `bohr lkm extract`，不是把文件发布到全局可搜索论文库；本地
知识库可立即使用抽取结果。相同内容的已验证文档可复用，不重复上传。

库的 `manifest.json` 只记录既有运行产物的引用、hash、处理版本和编译后的集成包。
论文事实仍由 Gaia 包保存，索引和 viewer 不成为第二份事实源。输入、代码、提示模板、
模型或编译器版本变化时，相应缓存不再复用；旧问题的候选答案不会进入新问题的证据。
并发更新会检测冲突，不覆盖另一轮更新；失败保留旧库和失败运行记录。

请保留 manifest 引用的运行目录；删除这些目录会使对应缓存失效。不同用户使用不同的
库目录。既有运行不会自动迁移或覆盖；省略 `--library`／`--documents` 就继续使用原来的
完整流程。`--papers` 仍可与 `--library` 配合，用明确选择的本地图／文本构建知识库。

full 模式不需要你手动选论文。检索阶段是：

1. **查询规划**：让模型把问题扩成多条简短英文 LKM 查询，覆盖核心概念、
   方法/系统、评估场景；英文问题和确定性关键词查询始终作为 fallback 保留。
2. **多查询检索**：最多执行 `retrieval.queries` 次 hybrid LKM 搜索
   （`conclusion,abstract` scope），按 LKM id 合并论文。
3. **多信源交叉验证**：把前 2 条计划查询同时发给 OpenAlex、Crossref、arXiv、
   Semantic Scholar 四个免费索引，按 DOI、arXiv id 或标题相似度记录每个候选
   被哪些独立信源确认（`corroboration`）。候选没出现在搜索结果里时，再用
   DOI 直接回查（最多 4 篇）。单个信源 429、超时或返回坏数据只记为不可用，
   不会中断检索。
4. **候选排序**：综合命中查询数、LKM rerank 分数、轻量词面相关性、独立信源
   确证数和（探测后的）图规模排序。
5. **图探测**：逐篇取论文图；空图、没有 conclusion 节点的图、过薄的图直接
   跳过。第一轮不够 `min_papers` 时会继续扩展探测，而不是整次失败。
6. **补建缺失图**：论文没有原生 LKM 图时，Ariadne 会依次从 arXiv、DOI 落地页
   （`citation_pdf_url`）、OpenAlex、Unpaywall、Semantic Scholar、Crossref 和
   MDPI CDN 解析 PDF；先下载 PDF，再并行跑最多 `retrieval.max_extractions` 个
   `bohr lkm extract`（3 个 worker）。闭源 PDF 和超过 LKM 50 页限制的 PDF 会
   记录为跳过，不消耗补图预算。每次成功 1.00 CNY（缓存命中 0.10 CNY），
   `--no-lkm-extract` 可关闭。
7. **选择与审计**：优先选有图、相关、且被 `retrieval.min_corroboration` 个
   独立源确认的论文；确认不足时逐级回退到「仅相关」再到全体有图候选，不会
   因为外部 API 限流而失败。把每条查询、候选、交叉验证结果、探测结果、跳过
   原因、相关性分数、提取任务和最终选择写入 `01_retrieval/selection.json`。

### 多信源交叉验证

LKM 是唯一的论文图来源，但单一索引会有漏检和元数据错配。Ariadne 因此把 LKM
当作**主索引**，把 OpenAlex / Crossref / arXiv / Semantic Scholar 当作
**独立证人**：被独立源确认的候选在排序和选择时优先。默认
`retrieval.min_corroboration = 1`，即「LKM + 至少 1 个独立源」；调到 2 就是
要求两个独立源同时确认。

- 每个候选的确认来源、匹配方式和状态写在 `selection.json` 的 `cross_check`
  字段，汇总在 `01_retrieval/sources.json`，原始返回按源保存在
  `01_retrieval/sources/<source>.json`。
- `--no-cross-check` 或 `retrieval.cross_check=false` 完全关闭这一步（离线、
  内网或信源不可达时用）。`retrieval.cross_check_sources` 选择参与的源，例如
  `openalex,crossref`。
- `--external-discovery`（配置 `retrieval.external_discovery`）把「至少 2 个
  独立源都返回、但 LKM 没返回」的论文加入候选池补召回。它们没有 LKM 原生图，
  因此和缺图论文一样走 PDF → `bohr lkm extract`，受 `retrieval.max_extractions`
  预算限制，会多花提取费用。
- 这些元数据 API 免费、只读，不产生 LKM 费用。Semantic Scholar 有限流，可以
  设置 `S2_API_KEY` 提高配额；Ariadne 默认只用前 2 条查询访问它们。

`--papers ...` 仍会完全绕过自动检索和交叉验证，适合你已经知道语料的情况。

调参：

```bash
./ariadne/run.sh ask "..." --queries 5 --candidate-limit 16
./ariadne/run.sh ask "..." --no-query-planner   # 只用确定性查询
./ariadne/run.sh ask "..." --no-cross-check     # 关闭多信源交叉验证
./ariadne/run.sh ask "..." --external-discovery # 用独立信源补充召回
./ariadne/run.sh config set retrieval.min_relevance 0.2
./ariadne/run.sh config set retrieval.min_corroboration 2
```

如果选出来的论文不对，先看 `selection.json`：`queries.json` 是查询计划，
`candidates.json` 是排序后的候选池，`search_raw_*.json` 是每次 LKM 原始响应，
`sources.json` 是各独立信源的查询与候选确认记录，`extract/*/submit.json` 和
`result.json` 是补建图的审计。选择会尽量补足 `--top-k`；相关性和独立确认
都是偏好，不是硬上限。

## 配置

### 文件位置与优先级

| 优先级 | 来源 |
| ---: | --- |
| 1 | CLI flags（`--api-key`、`--base-url`、`--model`、`--bohr-profile` 等） |
| 2 | 环境变量 |
| 3 | Ariadne 配置文件 |
| 4 | `agent/.env`（旧版，本地） |
| 5 | 内置默认值 |

配置文件按以下顺序解析：

```text
$ARIADNE_CONFIG
$XDG_CONFIG_HOME/ariadne/config.json
~/.config/ariadne/config.json
```

Windows 下如果设置了 `APPDATA`，使用 `%APPDATA%\ariadne\config.json`。

最小配置如下；完整示例见 [`config.example.json`](config.example.json)：

```json
{
  "version": 1,
  "model": {
    "provider": "deepseek",
    "api_key": "sk-...",
    "base_url": "https://api.deepseek.com/v1",
    "model": "deepseek-v4-flash",
    "merge_model": "deepseek-v4-flash"
  },
  "bohr": {
    "profile": "ariadne",
    "access_key": "...",
    "host": "https://open.bohrium.com",
    "auto_approve_spend": false,
    "payment_source": ""
  },
  "runtime": {
    "python": ""
  }
}
```

配置文件是**稀疏**的：只写你显式设置过的值。未设置的值走默认值，或继续向
环境变量 / `agent/.env` 回退。

### 配置命令

```bash
./ariadne/run.sh config path                 # 打印配置路径
./ariadne/run.sh config init                 # 交互式配置
./ariadne/run.sh config init --no-input ...  # 脚本化配置
./ariadne/run.sh config show                 # 生效配置，secret 打码
./ariadne/run.sh config show --json
./ariadne/run.sh config get model.model
./ariadne/run.sh config set model.model deepseek-v4-flash
./ariadne/run.sh config unset model.model
./ariadne/run.sh config test --api           # 发一次很小的 chat completion
./ariadne/run.sh config test --bohr          # bohr auth status
```

支持的 key：

| Key | 含义 |
| --- | --- |
| `model.provider` | `deepseek` 或 `openai-compatible` |
| `model.api_key` | 模型 key（输出会打码） |
| `model.base_url` | OpenAI-compatible base URL，默认 `https://api.deepseek.com/v1` |
| `model.model` | paper2ir / single 使用的模型 |
| `model.merge_model` | merge / Step 6 使用的模型 |
| `bohr.profile` | Bohr CLI profile；空表示 active profile |
| `bohr.access_key` | Bohrium access key（输出会打码） |
| `bohr.host` | 默认 `https://open.bohrium.com` |
| `bohr.auto_approve_spend` | true 时总是给计费 LKM 命令加 `--yes` |
| `bohr.payment_source` | 空（网关默认）、`photon` 或 `balance` |
| `runtime.python` | pipeline 子进程使用的 Python |
| `retrieval.use_query_planner` | 是否使用 LLM 查询规划器（默认 true） |
| `retrieval.queries` | 每个问题的最大 LKM 查询数（默认 5） |
| `retrieval.per_query` | 每条查询请求的结果数（默认 8） |
| `retrieval.candidate_limit` | 第一轮图探测数（默认 16，不够会扩展） |
| `retrieval.min_papers` | 至少需要几篇有图的论文（默认 2） |
| `retrieval.min_graph_nodes` | 低于该节点数的图视为过薄（默认 3） |
| `retrieval.min_relevance` | 优先论文的词面相关性门槛（默认 0.2） |
| `retrieval.use_lkm_extract` | 是否用 `bohr lkm extract` 补建缺失图（默认 true） |
| `retrieval.max_extractions` | 每次运行最多补建几篇（默认 5，0 关闭） |
| `retrieval.cross_check` | 是否用独立信源交叉验证候选（默认 true） |
| `retrieval.cross_check_sources` | 参与验证的源，逗号分隔（默认全部四个） |
| `retrieval.min_corroboration` | 优先选择需要的独立源数（默认 1，0 关闭门槛） |
| `retrieval.external_discovery` | 是否把多源一致但 LKM 漏检的论文加入候选（默认 false） |
| `retrieval.external_candidates` | 每次最多补充几篇外部论文（默认 4） |

### 环境变量

| 变量 | 用途 |
| --- | --- |
| `ARIADNE_CONFIG` | 配置文件路径 |
| `ARIADNE_MODEL_API_KEY` / `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` | 模型 key |
| `ARIADNE_MODEL_BASE_URL` / `DEEPSEEK_BASE_URL` / `OPENAI_BASE_URL` | endpoint |
| `ARIADNE_MODEL` / `DEEPSEEK_MODEL` / `OPENAI_MODEL` | 模型名 |
| `ARIADNE_MERGE_MODEL` / `DEEPSEEK_MERGE_MODEL` | merge 模型 |
| `GAIA_RELATION_CLASSIFIER` | 单篇 Step 3 四分类：`deepseek`（默认）或 `jev` |
| `TYPESAFE_API_KEY` | Jev 的 TypeSafe API key，仅用于关系类型分类 |
| `TYPESAFE_MODEL` | Jev 模型，默认固定为 `jev-1.14.0` |
| `ARIADNE_BOHR_ACCESS_KEY` / `BOHR_ACCESS_KEY` / `LKM_ACCESS_KEY` | Bohr key |
| `ARIADNE_BOHR_PROFILE` | Bohr profile |
| `ARIADNE_AUTO_APPROVE_SPEND` | `1` / `true` 表示自动批准计费调用 |
| `ARIADNE_PAYMENT_SOURCE` | `photon` 或 `balance` |
| `ARIADNE_PYTHON` | Ariadne 及其子进程使用的解释器 |
| `S2_API_KEY` | 可选的 Semantic Scholar key，提高交叉验证的限流额度 |

环境变量优先于配置文件，所以 CI 可以完全不写配置文件：

```bash
DEEPSEEK_API_KEY=... BOHR_ACCESS_KEY=... \
  ./ariadne/run.sh ask "..." --top-k 2 --yes
```

要用 Jev 分类，在本地 `agent/.env` 设置 `GAIA_RELATION_CLASSIFIER=jev`、
`TYPESAFE_MODEL=jev-1.14.0` 和 `TYPESAFE_API_KEY`，并将文件权限设为 `0600`。
该文件已被 Git 忽略，密钥不要提交。Jev 只替换单篇 Step 3 的
`deduction / abduction / analogy / null` 类型判断；保留导入关系的端点和顺序，
同时为原先未分类的可信导入关系补做类型判断。候选关系规范化、推理展开及跨论文
联合分析仍使用原来的生成模型。Jev 的原始概率和实际模型版本记录在既有工具审计中，
不写为图上的先验概率，也不引入未经验证的置信度阈值。
缺少密钥、鉴权失败或响应不合法时，该步骤明确失败，不静默回退为 DeepSeek 或 `null`。
已有运行不会自动重算；更换模型后，从需要重新分类的单篇 Step 2 检查点建立恢复运行。

### 模型兼容性

paper2ir、single、merge 使用 **OpenAI-compatible chat-completions** 客户端。
DeepSeek 是默认值，也是实际测试过的路径。只要 endpoint 支持
`POST {base_url}/chat/completions` 和
`Authorization: Bearer <key>`，就可以通过 `model.base_url`、`model.model`、
`model.merge_model` 接进来。

只设置 `ANTHROPIC_API_KEY` **不能**跑 full：single 和 merge 阶段需要
OpenAI-compatible 客户端。`doctor` 会明确提示这一点。

### Bohr 账户与计费

`bohr` 使用 profile。Ariadne 读 `bohr.profile`；为空时使用 active profile。
`ariadne auth bohr` 会创建专用 profile `ariadne`：

```bash
# access key（非交互）
./ariadne/run.sh auth bohr --access-key "$BOHR_ACCESS_KEY"

# 浏览器流程
./ariadne/run.sh auth bohr

# 无头机器 device flow
./ariadne/run.sh auth bohr --device

# 只检查
./ariadne/run.sh auth bohr --check
./ariadne/run.sh auth status
```

等价的 Bohr CLI 命令：

```bash
bohr auth login --ak "$BOHR_ACCESS_KEY" --ak-only --profile ariadne
```

LKM `search`、`graph`、`reasoning` 等调用按次计费（写作时 **0.05 CNY/次**，
个人每月 1,000 次免费额度）。PDF 解析单独计费。Ariadne 在第一次计费调用前
会询问；`--yes` 或 `bohr.auto_approve_spend=true` 表示授权本次运行。
`--dry-run` 不花钱。查询当前价格：

```bash
bohr billing pricing --resource lkm -o json
```

### Python 解释器

`run.sh` 在启动 Ariadne 之前就选好解释器；Ariadne 启动后，
`runtime.python` / `ARIADNE_PYTHON` 决定 pipeline 子进程用哪个 Python。
venv 不叫 `.venv` 时建议显式指定：

```bash
./ariadne/run.sh config set runtime.python /opt/venvs/gaia/bin/python
```

## 命令

| 命令 | 作用 |
| --- | --- |
| `ask "<question>"` | full：问题 → LKM → paper2ir → single → merge → render |
| `ask ... --mode demo` | 在冻结飞机图上跑答案层 |
| `ask ... --dry-run` | 预检并打印计划，不写文件、不花钱 |
| `ask ... --papers spec.json` | 用离线论文 spec 替代 LKM 检索 |
| `ask ... --top-k N` | 选出的论文数（先用 2） |
| `ask ... --queries N` | 自动检索的最大 LKM 查询数 |
| `ask ... --candidate-limit N` | 第一轮图探测的候选数 |
| `ask ... --no-query-planner` | 关闭 LLM 查询规划器，只用 fallback 查询 |
| `ask ... --no-lkm-extract` | 不用 `bohr lkm extract` 补建图 |
| `ask ... --max-extractions N` | 本次运行最多补建几篇 |
| `ask ... --no-cross-check` | 关闭多信源交叉验证 |
| `ask ... --min-corroboration N` | 优先选择需要的独立信源数 |
| `ask ... --external-discovery` | 用独立信源补充 LKM 漏检的论文 |
| `ask ... --yes` | 授权本次计费 LKM 调用 |
| `ask ... --no-fulltext` | 跳过 PDF 解析，改用图派生文本 |
| `demo` | 离线飞机示例 |
| `doctor` | 检查配置、Python、模型、Bohr、pipeline、就绪状态 |
| `config init/show/set/get/unset/path/test` | 管理配置 |
| `auth bohr/status` | Bohr 登录与认证状态 |

运行级 flags：

```text
--config PATH
--api-key KEY  --base-url URL  --model NAME  --merge-model NAME
--bohr-profile P  --bohr-access-key AK  --payment-source photon|balance
--python PATH
```

## 输出

每次 full / demo 运行都隔离在
`outputs/ariadne_runs/<timestamp>_<slug>/`：

| 文件 | 含义 |
| --- | --- |
| `domain_viewer_question.html` | 自包含问题图，无需服务器 |
| `ALL_CHAINS_TO_ROOTS.md` | 到达 R1–R5 的全部多步链，含 Mermaid 和内容表 |
| `05_outputs/answer_layer.json` | roots、links、类型化边、exploratory paths（full） |
| `05_outputs/domain_view_question.json` | 注入答案层后的 view model（full） |
| `run.json` | 问题、各阶段耗时、配置摘要（不含 secret）、输出路径 |
| `logs/` | 各阶段命令日志 |

离线 demo 的答案层和 view model 在 `04_question/`，不是 `05_outputs/`；两个
主交付物仍会复制到 run 根目录。

full 运行会保留中间阶段：

```text
01_retrieval/   queries.json、candidates.json、selection.json、sources.json、
                sources/<source>.json、search_raw_*.json、
                extract/<paper>/submit.json|result.json、每篇 graph.json + ocr.md
02_paper2ir/    run-local clean_claims 副本、每篇 claims_final.json
03_single/      Gaia IR package、formalization.json、knowledge.index.json
04_merge/       merge run、view_model.json
05_outputs/     答案层与交付物
```

`02_paper2ir/pipeline` 是每次运行单独复制的，所以不会复用旧数据，也不会让
上一次失败的 paper2ir 污染下一次运行。

## 工作原理

| 阶段 | 做什么 | 实现 |
| --- | --- | --- |
| `01_retrieval` | 规划多条 LKM 查询、合并排序候选、用独立信源交叉验证、探测论文图、跳过空图/无 conclusion/过薄图，再取全文或用图派生文本 | [`ariadne/retrieval.py`](ariadne/retrieval.py)、[`ariadne/sources.py`](ariadne/sources.py) |
| `02_paper2ir` | 14 步 paper2ir：LKM 图 + 文本 → `claims_final.json` | [`pipelines/clean_claims`](../pipelines/clean_claims/)（run-local 副本） |
| `03_single` | 维护中的 single pipeline：claims → `gaia.ir.json`、`formalization.json`、`knowledge.index.json` | [`pipelines/single_paper`](../pipelines/single_paper/) + harness |
| `04_merge` | 把多篇 package 合成一张领域图，含 K conclusions 和 Step 6 答案 root/link | [`pipelines/merge/pipeline.question.json`](../pipelines/merge/pipeline.question.json) |
| `05_render` | 渲染问题 viewer、导出全部链条 | [`render_question_viewer.py`](../pipelines/merge/pipeline_merge_question/render_question_viewer.py)、[`export_chains_markdown.py`](../pipelines/merge/pipeline_merge_question/export_chains_markdown.py) |

Ariadne 封装的三套维护中 pipeline：

1. **paper2ir** — 论文 → claims；
2. **single** — claims → 单篇推理图；
3. **merge** — 多图 → 领域图 + 问题答案层。

检索在三套 pipeline 之前；渲染在之后。

### “support”到底是什么意思

- 只有 **deduction** 边可以成链。`infer`、`analogy`、`bounds`、
  `special_case`、`same-paper` 会保留为可见上下文，但不会被升级为 deduction。
- `same-paper` 是按论文和 claim 编号排列的直接支持 claim；它是阅读顺序，
  不是逻辑证明。
- viewer 默认显示全部图路径。能画出来不等于有证据；链条导出器会标出边类型。
- root 是**候选答案**，不是真值标签。这张图是证据与缺口的分布图，不是裁决。

## 离线论文 spec

`--papers` 跳过 LKM 检索。spec 可以指向 LKM id，也可以指向本地文件：

```json
{
  "papers": [
    {
      "lkm_id": "1234567890123456789",
      "title": "Paper title",
      "arxiv": "2501.00001",
      "graph_path": "path/to/graph.json",
      "ocr_path": "path/to/paper.md"
    },
    {
      "title": "Local graph without an LKM id",
      "graph_path": "path/to/another-graph.json",
      "ocr_path": "path/to/another-paper.md"
    }
  ]
}
```

路径默认相对仓库根目录，也可以写绝对路径。接受的图结构：

- `{"code": 0, "data": {"papers": [...]}}`（pipeline 原生）；
- `{"data": {"items": [...]}}`（Bohr CLI 响应）；
- 裸 `{"nodes": [...], "edges": [...]}` 图；
- 单个带 `graph` / `paper` 的 paper item。

没有 LKM id 的本地图会得到稳定的 `local_<hash>` id，pipeline 仍可继续。

```bash
./ariadne/run.sh ask "你的问题" \
  --papers my-papers.json --top-k 2 --yes
```

本地 graph/text 条目不会花 LKM 调用；只有 `lkm_id` 的条目仍会调
`bohr lkm graph`，需要 `--yes`。

## 排障

| 现象 | 处理 |
| --- | --- |
| `bohr: command not found` | `npm install -g bohr-cli`，或用 `--papers` 本地文件 |
| `profile "ariadne" does not exist` | `./ariadne/run.sh auth bohr`，或把 `bohr.profile` 改成已有 profile |
| `bohr lkm search failed` / `401` | 检查 access key，跑 `ariadne auth bohr --check` 或 `bohr auth login` |
| `CONFIRMATION_REQUIRED` | 加 `--yes`，或设 `bohr.auto_approve_spend=true` |
| `No model API key is configured` | `config init`、`--api-key` 或 `DEEPSEEK_API_KEY` |
| 只设了 `ANTHROPIC_API_KEY` 但 full 失败 | 换 DeepSeek/OpenAI-compatible key；single/merge 需要它 |
| `ModuleNotFoundError: gaia` | `pip install --pre -r ariadne/requirements.txt` |
| `No module named pipeline_harness` | `pip install -e agent`，或用 `./ariadne/run.sh` |
| `--papers contains no usable entries` | 每条至少要有 `lkm_id` 或 `graph_path` |
| `Ariadne needs at least two papers` | 增大 `--top-k` 或给 spec 加论文 |
| `paper2ir stopped at ...` | 看 `02_paper2ir/logs/paper2ir_<id>.log`；run-local 数据会保留 |
| demo 正常但 full 很慢 | 先用 `--top-k 2 --no-fulltext`；paper2ir 每篇 14 个 LLM 步骤 |
| Windows 路径错误 | 在仓库内建 venv，从仓库根目录运行；显式设 `runtime.python` |

## 目录

| 路径 | 用途 |
| --- | --- |
| [`ariadne/`](.) | 本产品：CLI、配置、检索、阶段编排、渲染 |
| [`agent/`](../agent/) | pipeline harness、artifact store、checkpoint、校验、viewer 源 |
| [`pipelines/clean_claims/`](../pipelines/clean_claims/) | paper2ir：论文图 → claims |
| [`pipelines/single_paper/`](../pipelines/single_paper/) | claims → 单篇 Gaia IR |
| [`pipelines/merge/`](../pipelines/merge/) | 多 package → 领域图 + 问题层 |
| [`weekly_reports/export_single_artifacts.py`](../weekly_reports/export_single_artifacts.py) | 导出 single 运行产物给 merge |
| [`outputs/aircraft/final/`](../outputs/aircraft/final/) | 冻结飞机 demo 数据 |
| [`outputs/physics/final/`](../outputs/physics/final/) | 复核版 Physics 示例图 |

## 成本

| 资源 | 成本 |
| --- | --- |
| LKM search / graph / reasoning | 0.05 CNY/次，个人每月 1,000 次免费额度 |
| LKM PDF 抽取 | 单独计费；用 `bohr billing pricing` 查 |
| LLM 调用 | 由你的 provider 计费；paper2ir 每篇 14 步，之后还有 single/merge |

两篇论文的自动运行最多产生 `retrieval.queries` 次 search 加图探测
（第一轮最多 `retrieval.candidate_limit` 次，图不够才扩展），然后才进入
LLM pipeline。缺图的论文最多补建 `retrieval.max_extractions` 次，每次成功
1.00 CNY（缓存命中 0.10 CNY）。多信源交叉验证只调用免费的元数据 API，
不产生 LKM 或模型费用；开启 `--external-discovery` 后补回来的论文才可能
消耗提取预算。用 `--dry-run` 看上限计划，用 `--no-fulltext` 跳过 PDF 解析。

## License

仓库目前没有 license 文件。请按研究原型使用；如果要作为库再分发，先补
license。
