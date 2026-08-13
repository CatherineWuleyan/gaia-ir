# LTH 粗推理图 weakpoint 标注

## 标注口径

本文件依据 `05-formalization-methodology.md` 的 Step 2，对当前 LTH 示例的粗推理图进行连接级标注。

- **weakpoint 是连接，不是批评节点。** 它表示从证据、有限实例或解释性假说到目标 claim 之间尚未展开的 `↝`。
- 现有 `P1–P5` 是独立的限制/质疑命题，可用于约束 weakpoint，但它们本身不是 weakpoint 连接。
- `H1–H8` 是观测或证据摘要，通常位于 weakpoint 的证据端。
- 实验定义、设置和必要条件应作为 premise；不能因为尚未入图就一律称为 weakpoint。
- 初始轮只识别位置和类型且不填写主观 `(p₁,p₂)`；Figure 1 的 `WC-09A/09B/10` 现已补入待审阅的 Step 3 子网络草稿。

廖悦辛在飞书中的表述与此一致：当前 `graph.json` 没有可靠的原始推理类型，需要根据内容重新识别；本论文的主要推理以实验产生的 abduction/induction 为主，且现有 clean claims 尚缺实验 claim。

## 已标注的连接

| ID | 结论 | 从什么推到什么 | 初步类型 | 主要问题 |
|---|---|---|---|---|
| `OTWTA-WC-01` | C01 | 被测源—目标迁移实验 → 自然图像域内跨数据集迁移 | induction | 有限数据集与两种架构不能严格蕴含任务族规律 |
| `OTWTA-WC-02` | C01 | 离散剪枝比例上的曲线 → 广泛比例及极高稀疏度下都可迁移 | induction | 对连续区间和架构临界点存在外推 |
| `OTWTA-WC-03` | C01/C02 | 源数据集排序 → 规模、类别数或复杂度预测迁移性 | induction + abduction | 多个数据集属性共同变化；与 `P1` 对应 |
| `OTWTA-WC-04` | C01 | Fashion-MNIST 性能/过拟合曲线 → 迁移彩票发挥正则化作用 | abduction | 正则化不是曲线唯一可能的因果解释 |
| `OTWTA-WC-05` | C02 | CIFAR-10/100 对比 → 类别数是独立预测因素 | abduction | 类别粒度、内容和任务难度未控制；与 `P1` 对应 |
| `OTWTA-WC-06` | C03 | VGG19 的 SGD↔Adam 实验 → VGG 彩票可跨优化器迁移 | induction | 单一架构且粗图未明确数据集；与 `P2` 对应 |
| `OTWTA-WC-07` | C04 | CIFAR-10a→10b 一次切分 → 同分布样本间可泛化 | induction | 单次确定性切分不能建立一般规律；与 `P3` 对应 |
| `OTWTA-WC-08` | C04 | 留出子集迁移成功 → 彩票不是记忆样本而是捕获分布级结构 | abduction | 排除了直接记忆，但没有唯一确定机制 |
| `OTWTA-WC-09` | C05 | CIFAR-10/VGG19 比较 → 全局剪枝的一般性能与逐层稀疏规律 | induction | 拆为 `09A`（性能）与 `09B`（逐层模式），各用单实例 abduction 单元展开；与 `P4` 对应 |
| `OTWTA-WC-10` | C05 机制归因 | O1 性能优势 + O2 逐层模式 → “浅层绝对参数过少损害表达能力” | abduction | 作者解释作为 H 与层敏感性、幅值分布、优化动态等 Alt 竞争；不建自然语言“解释”边，仍需机制消融 |
| `OTWTA-WC-11` | C06 | 掩码置换性能差异 → 掩码结构包含有用信息 | abduction + induction | 因果归因与单配置推广；与 `P5` 对应；`R009` 不应标为演绎 |
| `OTWTA-WC-12` | C06 | 掩码携带信息 → 公平基线必须全局置换掩码 | premise gap | 缺少“公平基线不得继承发现信息”的显式前提 |
| `OTWTA-WC-13` | C07 | 特定日程成功 → 该日程是大型场景中的有效流程 | scope-sensitive | 描述“论文采用”可严格支持；主张“一般有效/机制正确”则是 induction/abduction |

详细的 source step、clean claim、桥梁命题和 Step 3 模式见 `lth_coarse_graph_weakpoints.json`。

## 现有粗图中需要纠正的三点

1. `P1–P5 -> factor` 的 `weakpoint_of` 命名混合了“限制命题”和“weakpoint 连接”两个概念。后续可保留 P 节点，但应把真正的 `↝` 单独建成连接节点。
2. `clean_claims_and_relations.json` 的 `R009` 把 `A021 -> A020` 标为演绎；从置换实验到“掩码结构包含信息”实际包含因果解释，应标为 abduction weakpoint。
3. C07 必须先固定目标语义。如果只记录实验方法，日程是 premise/事实；如果主张该日程有效、可推广或解释成功原因，就必须保留 `OTWTA-WC-13`。

## 进入步骤二时的优先顺序

先补能直接落到实验图表的 observation claim，再展开 weakpoint：

1. `OTWTA-WC-11/12`：Fig. A1 的掩码置换实验，边界最清楚。
2. `OTWTA-WC-09A/09B/10`：Fig. 1 的两条归纳规律及作者机制归因；`WC-10` 用独立 abduction 子网络展开。
3. `OTWTA-WC-07/08`：Fig. 2 的 CIFAR-10a/b 留出实验。
4. `OTWTA-WC-06`：Fig. 5 的跨优化器实验。
5. `OTWTA-WC-01–05`：跨数据集综合结果，需要拆分最多的实验单元。
6. `OTWTA-WC-13`：先确认 C07 是描述性 claim 还是有效性 claim。
