# Cross-paper relation audit

日期：2026-10-18

## 原则

- 跨论文关系全部保留，默认显示，不隐藏。
- 只在关系确实符合时使用具体类型；否则保留 `infer`，并标为弱关系。
- 跨飞机/跨框架的方法迁移用 `analogy`；`analogy` / `infer` 都不进入 support chain。
- 没有把跨论文关系强行升级成 `deduction`。

## 结果

- 跨论文边：101
- 改类型：54
- 最终类型分布：`analogy` 52, `bounds` 4, `conjunction` 18, `infer` 26, `special_case` 1
- chainable：0（没有跨论文 deduction）

## 主要修正

1. 3 条 `reproduces` 全部修正：2 条改为 `analogy`（不同飞机/不同量），1 条改为 `infer/enabling`（模型系数被方程使用，不是独立复现）。
2. 12 条跨飞机/跨框架的 `special_case` 改为 `analogy`；只保留 1 条真正的数学特例（`P303→P200`，稳态点质量平衡是点质量方程的特例）。
3. 1 条把“速度范围”误标成 `bounds` 的边改为 `infer`；其余 4 条数值/范围 bound 保留，但加跨对象 caveat。
4. 38 条明显的跨飞机/跨框架 `infer` 改为 `analogy`；其余 26 条保留 `infer`，作为“方向清楚但没有更强类型”的弱关系。
5. 18 条 `conjunction` 保留为对称算子，不算 reasoning type。

## 逐条记录

每条边的当前类型、最终类型、是否修改、理由见 `cross_paper_audit.json`。

## 对图的影响

- 跨论文边默认可见；`analogy` 为紫色虚线，`infer` 为灰色虚线，`bounds` 为橙色虚线，`conjunction` 为紫色虚线。
- 仍然没有 typed support chain：这些都是跨论文关系，不是同对象 deduction。
- same-paper direct-support 序列继续作为同论文阅读顺序；跨论文关系只作为关系网络，不作为 support。
