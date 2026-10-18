# 问题主导飞机领域图：实现与效果评估

日期：2026-10-17
范围：`pipelines/merge/pipeline_merge_question`（新副本）与
`outputs/question_domain_aircraft`（本次冻结图运行结果）

## 0. 结论摘要

- 新增了 `pipeline_merge_question` 副本，原 `pipeline_merge` 未改。
- 新增 Step 6：问题 → 3–5 个候选回答（根节点）→ 在冻结候选池里连接 claim →
  预计算 premise-to-answer 链。
- 在**当前 311 claim / 208 edge / 13 paper 的冻结飞机图**上跑通：
  - 5 个候选回答；
  - 50 条 answer edge（45 support + 5 bounds，0 oppose）；
  - 46 条 claim 被直接连接，86 条进入连接或链，占 connected claims 的 36.5%；
  - 12/13 篇论文被触及；
  - 6 次 LLM 调用，约 53.6k tokens。
- 五个回答覆盖了不同的工作流阶段：集成概念设计环境、嵌套 MDO、非线性飞行
  动力学建模、飞行数据模型校正、硬件在环验证。
- 当前冻结图**没有 K（领域结论）层**；本次回答直接连 claim。完整 pipeline
  配置已把 `summarize_conclusions` 改为 `true`，下次完整 run 会先产出 K，
  Step 6 可以优先把回答连到 K。
- 主要不足：所有根都是 `primary`（2 篇论文直接支持），没有 `contested`；
  直接支持仍偏向每篇论文的 dominant evidence；候选池有界，连接召回不是全量。

## 1. 上一次为什么没有领域结论层

本次输入 `weekly_reports/out/domain_graph_aircraft_reviewed/source_snapshot/`
的 `merge_summary.json` 明确记录了：

```json
"skip_domain_conclusion_summary": true
```

对应的旧配置是：

```json
"step3_identify_structures": {
  "options": { "summarize_conclusions": false, "allow_infer": true }
}
```

因此当前 view 的 311 个节点全部是 P claim；`metadata.label_scheme` 的
`by_kind` 只有 `{"P": 311}`，没有 K。新副本的 `pipeline.question.json` 已改为
`summarize_conclusions: true`，并保留 `allow_infer: true`。

## 2. 本次改动

| 层 | 文件 | 说明 |
| --- | --- | --- |
| 算法 | `pipelines/merge/pipeline_merge_question/question_layer.py` | 根生成、连接、链、注入、评估 |
| Pipeline | `.../step6_question_layer.py` | 新增 Step 6，产出 `question.answer_layer` |
| View | `.../question_view.py` | 把回答层注入 merge view |
| Config | `pipelines/merge/pipeline.question.json` | K 层 + Step 6 + 新 view adapter |
| Runner | `.../run_question_layer.py` | 只对冻结 view 跑回答层 |
| Viewer | `.../viewer_question.html` | 根节点/回答边/回答层开关/根节点详情 |
| Render | `.../render_question_viewer.py` | 把任意 view model 注入 viewer |

原 `pipeline_merge` 的 Step 0–5 代码没有修改；副本与它是并列目录。

## 3. 运行设置

- 输入图：`weekly_reports/out/domain_graph_aircraft_reviewed/source_snapshot/domain_view_model.json`
  - 311 claims，208 edges，13 papers，222 connected claims，89 isolated claims。
- 问题：`怎么完成飞机的仿真设计？`
- 模型：`deepseek-v4-flash`（`agent/.env`，Step 3 同一调用通道）。
- 根数量上限：5。
- 输出：`outputs/question_domain_aircraft/`。
- Viewer：`outputs/question_domain_aircraft/domain_viewer_question.html`。
- 预览：`outputs/question_domain_aircraft/domain_viewer_question.preview.png`。

> 本次实际执行的是 `run_question_layer.py`，即对冻结 311 claim 图只跑 Step 6
> 回答层。完整 `pipeline.question.json` 已通过 `pipeline_harness init` / `check`，
> 但**没有**执行 Step 0–6 的完整 LLM run（会重新跑 Step 2/3，成本高）；含 K 层
> 的完整结果留待下一次实验。

## 4. 生成结果

### 4.1 五个根

| 根 | 标题 | 类型 | support | papers | chains | leaves | depth | bounds |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R1 | 集成概念设计环境 | method_route | 10 | 2 | 22 | 14 | 4 | 0 |
| R2 | 嵌套多学科优化 | method_route | 8 | 2 | 26 | 13 | 5 | 2 |
| R3 | 非线性飞行动力学建模 | modeling_choice | 8 | 2 | 27 | 16 | 9 | 2 |
| R4 | 飞行数据模型校正 | method_route | 10 | 2 | 24 | 13 | 6 | 0 |
| R5 | 硬件在环验证 | validation_route | 9 | 2 | 19 | 13 | 7 | 1 |

- R1 直接支持来自 `paper_aec49f8f`（CACDS/ModelCenter 集成）和
  `paper_c6f44cb4`（模块化 MDO）。
- R2 直接支持来自 `paper_9c2fb427`（结构/静稳定约束）和 `paper_c6f44cb4`
  （嵌套 MDO）。
- R3 直接支持来自 `paper_014abc68`（35 方程 DAE 飞行动力学）和
  `paper_26dddb44`（实时仿真）。
- R4 直接支持来自 `paper_066432fe`（delta 气动模型）和 `paper_53166ac1`
  （模拟双脉冲参数辨识）。
- R5 直接支持来自 `paper_26dddb44`（联合 HIL）和 `paper_755906eb`
  （Paparazzi SITL/HITL 工作流）。

### 4.2 链与覆盖

- 45 条 support、5 条 bounds、0 条 oppose。
- 46 条 claim 被直接连接；83 条非根 claim 出现在链上；去重后 86 条被覆盖
  （connected claims 的 36.5%，全部 claims 的 27.7%）。
- 12/13 篇论文被触及；未触及的是 `paper_0fc191d1`（18 claims，主要是
  CZ-150 的飞行动力学/气动建模细节）。
- 4 条 claim 同时支持两个根：
  - `paper_c6f44cb4::claim_1/5/25`（MDO 架构）同时支持 R1/R2；
  - `paper_26dddb44::claim_4`（实时仿真计算动力学）同时支持 R3/R5。
- 链最深的是 R3（depth 9，16 个 leaf），最浅的是 R1（depth 4）。

### 4.3 与手工 reviewed R1–R5 的对照

手工 reviewed 版的 R1–R5 是**案例结果**（CJX 成本、滑翔机预测、CZ-150 NRMSE、
NASA 重量、Minion HIL），不是方法路线。本次生成的是**方法路线**，两者层级不同，
可以互补：

| 新 root | 对应手工 R |
| --- | --- |
| R1 集成概念设计环境 | 手工 R1（CJX 案例） + MDO 相关证据 |
| R2 嵌套多学科优化 | 手工 R4（NASA 静稳定/重量） |
| R3 非线性飞行动力学建模 | 手工版没有对应根（新路线） |
| R4 飞行数据模型校正 | 手工 R2（滑翔机预测） |
| R5 硬件在环验证 | 手工 R5（Minion HIL） |

手工版的 R1–R5 是“已有事实命题的显示角色”；本次的 R1–R5 是“针对问题的候选
回答”，支持关系是新加的回答边。两者不能直接按节点数比较。

## 5. 质量评估

### 做得好的地方

1. **根之间有清晰的工作流分工**：设计环境、优化、建模、校正、验证，不是同一篇
   论文换一种说法。
2. **每个根都有跨论文直接支持**：5 个根都是 2 篇论文；R1/R2 通过 MDO 架构共享
   证据，R3/R5 通过实时仿真共享证据，说明回答层利用了 integration delta。
3. **链是跨论文的**：例如 R2 的链从 `paper_aec49f8f` 出发，经过
   `paper_c6f44cb4`，最后落到 `paper_9c2fb427` 的静稳定证据；R4 的链跨
   `paper_6a2ef68a → paper_18f90780 → paper_066432fe`。这不是单篇论文内部的
   伪链。
4. **边界被显式记录**：R2 记录“该 MDO 问题省略了应力/位移约束”“位移约束不
   活跃”；R3 记录“逆仿真输入模式”“左右对称飞机的化简”；R5 记录“HIL 与飞行
   约有 10% 差异”。这些是回答的适用条件，不是支持票。
5. **问题在图上可见**：viewer 标题、图例、默认 answer-focused 模式、根节点详情都直接
   围绕问题，读者不需要先浏览 311 条 claim；需要看全图时勾选 `Full domain graph`。

### 需要警惕的地方

1. **所有根都是 `primary`**：因为判定规则是“≥3 support 且 ≥2 paper”，5 个根
   都满足。这个标签只说明“有多篇论文的直接支持”，不说明回答正确，也不区分
   支持强度。R1/R4 有 10 条 support，R3 只有 8 条，但 verdict 相同。
2. **没有 oppose**：当前图只有 infer/conjunction，没有 contradiction；LLM 也没
   找到可辩护的反对关系。这与 reviewed 物理图里 `contested` 的候选不同，不能
   把“没有 oppose”读成“回答无争议”。
3. **直接支持偏向 dominant paper**：每个根的直接支持大多集中在 1 篇主论文，
   第二篇通常只有 1–2 条。跨论文性主要出现在链上；如果只看根的直接 support，
   仍可能高估“多论文共识”。
4. **候选池有界**：每个根只把 44–60 条 claim 交给连接判断；池外相关 claim 会被
   判 none。这是成本控制，也意味着 36.5% 覆盖率不是“全图只有 36.5% 相关”，
   而是“本轮明确连上的比例”。
5. **提示词敏感**：第一版（保存在 `outputs/question_domain_aircraft_v1/`）生成
   的 5 个根偏向较小的论文簇（飞行数据、嵌套 MDO、HIL、data-enhanced
   simulation、低成本 SITL/HIL），漏掉最大的集成设计环境和飞行动力学簇，且
   HIL 出现两次。调整 digest/prompt（每篇论文一个代表 + 工作流阶段覆盖要求）
   后才得到现在的平衡集合。这说明根集合不是唯一解。
6. **LLM 连接未经逐条原文核验**：本次抽查了 15 条 support/bounds（重点看
   R1/R2/R5），未发现明显“只是主题相关”的连接；但没有对 50 条逐条对照原文。
   生产使用应加人工审核。
7. **Viewer 是原型**：默认 answer-focused 模式在 1600px 下布局横向，根节点下方留白
   较多。图例里的 `Full domain graph` 默认不勾选；勾选后显示 232 个 connected
   domain nodes（含未连到 R 的节点），布局更散，但能核对领域全图。

## 6. 成本

本次 6 次 LLM 调用（1 次根生成 + 5 次连接），共 53,587 tokens（prompt 33,543，
completion 20,044）。没有触发重试。`usage.ndjson` 有逐次记录。

## 7. 复现

冻结图（本次）：

```bash
python3 pipelines/merge/pipeline_merge_question/run_question_layer.py \
  --view weekly_reports/out/domain_graph_aircraft_reviewed/source_snapshot/domain_view_model.json \
  --out outputs/question_domain_aircraft
```

离线重放（不调用 LLM）：

```bash
python3 pipelines/merge/pipeline_merge_question/run_question_layer.py \
  --view weekly_reports/out/domain_graph_aircraft_reviewed/source_snapshot/domain_view_model.json \
  --fixture outputs/question_domain_aircraft/answer_fixture.json \
  --out outputs/question_domain_aircraft_replay
```

完整 pipeline（含 K 层）见
`pipelines/merge/pipeline_merge_question/README.md` 第 4 节。

## 8. 后续优先级

1. **先跑一次含 K 的完整 pipeline**，比较“回答→K→claim”与“回答→claim”两种
   连接的支持质量；这是最直接验证“跳过领域结论层”代价的实验。
2. **人工审核根节点和连接**，把确认结果写成 fixture，保证跨 run 稳定身份。
3. **补 oppose/contradiction 来源**：当前没有 contested 根，verdict 信息量有限。
4. **扩大连接召回**：第二轮候选池 + 人工标注集，评估 precision/recall。
5. **改根节点布局**：根带 + 链列，减少横向铺开和底部留白。
6. **把回答层 artifact 纳入 run 的版本管理**：question、prompt_version、model、
   fixture/hash 一起冻结，保证可审计。

## 9. 链级复核（新增）

逐根典型支持链、每步边类型（dependency / enabling / instance / analogy /
independent reproduction）、人工复核 action 与可靠性分级，见
`outputs/question_domain_aircraft/CHAIN_RELIABILITY.md`；
机器可读的链/边/claim 数据见 `outputs/question_domain_aircraft/typical_chains.json`。
