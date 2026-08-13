# Gaia-IR 工作资料清单

整理日期：2026-08-10

本目录保存了当前对话中出现的全部可下载资料，共 19 个源文件，分为论文样例、LKM 参考资料和 Gaia 官方文档三类。另以本文件作为索引。

## 目录结构

```text
gaia-ir/
├── DOCUMENT_INDEX.md
├── 01_paper_sample_one_ticket/
├── 02_lkm_reference/
└── 03_gaia_official_docs/
    ├── theory/
    └── gaia-ir/
```

## 1. 论文样例

目录：[`01_paper_sample_one_ticket/`](01_paper_sample_one_ticket/)

对应论文：*One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers*。

| 文件 | 功能 |
|---|---|
| [`paper_text_ocr_raw.md`](01_paper_sample_one_ticket/paper_text_ocr_raw.md) | 从论文生成的较原始 OCR 文本。用于回查论文正文、章节、图注和实验描述。 |
| [`paper_text_ocr_cleaned.md`](01_paper_sample_one_ticket/paper_text_ocr_cleaned.md) | 转换整理后的 OCR 文本，排版更适合日常阅读；遇到内容缺失时回查 raw 版本。 |
| [`clean_claims_and_relations.json`](01_paper_sample_one_ticket/clean_claims_and_relations.json) | 清洗并拆分后的命题与部分关系。包含 23 个 assertions 和 11 个 relations，是生成 Gaia-IR claim 的主要候选输入。 |
| [`lkm_coarse_reasoning_graph.json`](01_paper_sample_one_ticket/lkm_coarse_reasoning_graph.json) | LKM 生成的论文级粗推理图。包含 conclusion、highlight、weak_point、reasoning_steps、subproblem 以及它们之间的边。 |
| [`lth_coarse_graph_weakpoints.json`](01_paper_sample_one_ticket/lth_coarse_graph_weakpoints.json) | 按 Gaia formalization Step 2 对当前 LTH 粗图所做的连接级 weakpoint 标注；包含 13 个待展开连接、来源步骤、桥梁命题、推理类型和关联限制。 |
| [`LTH_WEAKPOINT_ANNOTATIONS.md`](01_paper_sample_one_ticket/LTH_WEAKPOINT_ANNOTATIONS.md) | 供人工复核的 weakpoint 清单，解释 P1–P5、H1–H8 与真正 weakpoint 连接的区别。 |
| [`figure1_experiment_claims_draft.json`](01_paper_sample_one_ticket/figure1_experiment_claims_draft.json) | Figure 1 的 S/A/B/M/R/U 结构化候选，包括两个核心实验 claim、两个可选随机基线 claim 和来源质量标记。 |
| [`FIGURE1_EXPERIMENT_CLAIMS_REVIEW.md`](01_paper_sample_one_ticket/FIGURE1_EXPERIMENT_CLAIMS_REVIEW.md) | Figure 1 实验 claim 的人工审阅稿，含 Mermaid 推理可视化和待确认问题。 |
| [`original_feishu_attachment_bundle.zip`](01_paper_sample_one_ticket/original_feishu_attachment_bundle.zip) | 廖悦辛发送的原始附件包，保留用于校验文件来源和恢复原始文件名。 |

## 2. LKM 参考资料

目录：[`02_lkm_reference/`](02_lkm_reference/)

| 文件 | 功能 |
|---|---|
| [`lth_100_papers_coarse_reasoning_graphs.zip`](02_lkm_reference/lth_100_papers_coarse_reasoning_graphs.zip) | 100 篇 LTH 论文的 LKM 粗推理图集合，用于观察粗图结构、测试转换流程和后续批处理。 |
| [`lkm_reasoning_graph_visualizer.html`](02_lkm_reference/lkm_reasoning_graph_visualizer.html) | 本地 HTML 可视化工具，用于打开和查看 LKM reasoning graph。 |
| [`reasoning_reduction_table_entailment_equivalence_contradiction.jpg`](02_lkm_reference/reasoning_reduction_table_entailment_equivalence_contradiction.jpg) | 聊天截图：展示 entailment、abduction、induction、analogy 等推理如何归约为 entailment、equivalence、contradiction 结构。 |

注意：`lth_100_papers_coarse_reasoning_graphs.zip` 内有 100 个归档条目，但条目路径都叫 `graph.json`。不要直接用普通“全部解压”方式写入同一目录，否则可能发生同名覆盖；应逐条读取归档 entry，或在导出时按论文 ID 分目录命名。

## 3. Gaia 官方理论文档

目录：[`03_gaia_official_docs/theory/`](03_gaia_official_docs/theory/)

| 文件 | 功能 |
|---|---|
| [`03-propositional-operators.md`](03_gaia_official_docs/theory/03-propositional-operators.md) | 定义命题算子、软蕴含、equivalence、contradiction 等基础语义。 |
| [`04-reasoning-strategies.md`](03_gaia_official_docs/theory/04-reasoning-strategies.md) | 定义演绎、溯因、归纳、类比、外推等推理策略及其微观结构。 |
| [`05-formalization-methodology.md`](03_gaia_official_docs/theory/05-formalization-methodology.md) | 从自然语言科学论证到命题网络的完整方法论；包含“提取命题—识别 weakpoint—细化推理”的示例。 |

## 4. Gaia-IR 官方结构文档

目录：[`03_gaia_official_docs/gaia-ir/`](03_gaia_official_docs/gaia-ir/)

| 文件 | 功能 |
|---|---|
| [`01-overview.md`](03_gaia_official_docs/gaia-ir/01-overview.md) | Gaia-IR 的目标、边界、设计原则和整体概念。 |
| [`02-gaia-ir.md`](03_gaia_official_docs/gaia-ir/02-gaia-ir.md) | 核心结构定义，包括 Knowledge、Operator、Strategy、Compose 和 graph。 |
| [`03-identity-and-hashing.md`](03_gaia_official_docs/gaia-ir/03-identity-and-hashing.md) | QID、对象身份、内容哈希及跨 package 引用规则。 |
| [`04-helper-claims.md`](03_gaia_official_docs/gaia-ir/04-helper-claims.md) | helper claim 的命名、作用域，以及 public/private 中间节点边界。 |
| [`05-canonicalization.md`](03_gaia_official_docs/gaia-ir/05-canonicalization.md) | 命题与图结构的规范化、去重及跨 package 对齐规则。 |
| [`06-parameterization.md`](03_gaia_official_docs/gaia-ir/06-parameterization.md) | claim prior、Strategy 概率参数及参数输入层的定义。 |
| [`07-lowering.md`](03_gaia_official_docs/gaia-ir/07-lowering.md) | Gaia-IR 如何被 backend 消费并转换成 runtime graph。 |
| [`08-validation.md`](03_gaia_official_docs/gaia-ir/08-validation.md) | 对象级和图级结构校验规则，是检查 IR 是否 contract-valid 的主要依据。 |

这些文件来自 `SiliconEinstein/Gaia` 官方 GitHub 仓库的 `main` 分支。部分文档标记为 `Target design`，因此实际使用时还应核对当前代码实现和 schema 版本。

## 5. 原文件名与新文件名

| 原名称 | 整理后名称 | 重命名原因 |
|---|---|---|
| `867771291879342656.zip` | `original_feishu_attachment_bundle.zip` | 数字 ID 无法体现它是飞书原始样例包。 |
| `ocr.md` | `paper_text_ocr_raw.md` | 明确这是论文原始 OCR 文本。 |
| `ocr_converted.md` | `paper_text_ocr_cleaned.md` | 明确这是整理后的 OCR 阅读版。 |
| `conclusions_final.json` | `clean_claims_and_relations.json` | 明确文件实际包含 assertions 和 relations，而不只是 conclusions。 |
| `graph.json` | `lkm_coarse_reasoning_graph.json` | 区分 LKM 粗图与后续 Gaia-IR 图。 |
| `graph.zip` | `lth_100_papers_coarse_reasoning_graphs.zip` | 明确是 100 篇 LTH 论文的粗图集合。 |
| `reasoning_graph_visualizer.html` | `lkm_reasoning_graph_visualizer.html` | 标明工具面向 LKM 粗推理图。 |
| 飞书图片资源 | `reasoning_reduction_table_entailment_equivalence_contradiction.jpg` | 根据截图实际内容和真实 JPEG 编码命名。 |

## 6. 推荐阅读顺序

1. `paper_text_ocr_cleaned.md`
2. `clean_claims_and_relations.json`
3. `lkm_coarse_reasoning_graph.json`
4. `theory/03-propositional-operators.md`
5. `theory/04-reasoning-strategies.md`
6. `theory/05-formalization-methodology.md`
7. `gaia-ir/01-overview.md`
8. `gaia-ir/02-gaia-ir.md`
9. `gaia-ir/04-helper-claims.md`
10. `gaia-ir/08-validation.md`
11. 需要实际编译、参数化或跨论文对齐时，再读其余 Gaia-IR 文档。

## 7. 覆盖范围

- 飞书身份：`user`（吴乐言）。
- 飞书范围：此前读取的吴乐言与廖悦辛私聊中的全部文件和图片附件。
- 已成功保存：原始论文样例包、100 篇粗图包、可视化 HTML、方法截图。
- GitHub 范围：本次对话中明确引用的 11 份 Gaia 官方文档。
- 未读取或未保存项：无。
- 外部写入：无；未发送消息，未修改飞书资源。
