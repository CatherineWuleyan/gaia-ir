# Question-led answer layer: evaluation

- Question (zh): 怎么完成飞机的仿真设计？
- Question (en): How can aircraft simulation design be carried out?
- Base graph: 311 claims, 223 edges, 13 papers, 226 connected claims
- Candidate answers: 5 roots, 50 answer edges
- Linked / chained claims: 46 (18.1% of connected claims, 14.8% of all claims)
- Papers touched: 8
- Claims supporting more than one answer: 4
- Chainable edges: 0; edge role counts: {'bounds': 5, 'conjunction': 18, 'infer': 184, 'reproduces': 3, 'special_case': 13}
- Typed support-chain coverage: 46 claims; with exploratory paths: 86 (35.8% of connected)

## Candidate answers

### R1 · 集成概念设计环境 / Integrated conceptual design environment

- Verdict: primary
- Support: 10 claims from 2 papers; typed chains: 0 (no_typed_chain; leaves 0, max depth 0)
- Exploratory (non-evidential) paths: 18 (20 nodes)
- Boundary: 0 · Oppose: 0
- Chain nodes: 0
- Statement: Aircraft simulation design can be organized around an integrated conceptual-design and simulation-optimization environment that runs an end-to-end civil-airplane design case and uses an optimizer to minimize a design objective such as direct operating cost.

### R2 · 嵌套多学科优化 / Nested multidisciplinary optimization

- Verdict: primary
- Support: 8 claims from 2 papers; typed chains: 0 (no_typed_chain; leaves 0, max depth 0)
- Exploratory (non-evidential) paths: 24 (23 nodes)
- Boundary: 2 · Oppose: 0
- Chain nodes: 0
- Statement: Conceptual aircraft design can be pursued through a unified modular multidisciplinary design optimization framework with nested optimization levels that optimize configuration and wing-box design variables, while structural and static-margin constraints may materially affect the resulting design.

### R3 · 非线性飞行动力学建模 / Nonlinear flight-dynamics formulation

- Verdict: primary
- Support: 8 claims from 2 papers; typed chains: 0 (no_typed_chain; leaves 0, max depth 0)
- Exploratory (non-evidential) paths: 24 (31 nodes)
- Boundary: 2 · Oppose: 0
- Chain nodes: 0
- Statement: Aircraft simulation can be formulated as a coupled nonlinear differential-algebraic flight-dynamics model with six-degree-of-freedom equations expressed in body or wind axes, requiring state variables and parameters to be specified.

### R4 · 飞行数据模型校正 / Flight-data model calibration

- Verdict: primary
- Support: 10 claims from 2 papers; typed chains: 0 (no_typed_chain; leaves 0, max depth 0)
- Exploratory (non-evidential) paths: 20 (24 nodes)
- Boundary: 0 · Oppose: 0
- Chain nodes: 0
- Statement: Simulation-model fidelity can be improved by identifying aerodynamic delta or parameter models from flight-test or simulated doublet data and updating the original aerodynamic database, followed by checking predictions against measured flight parameters.

### R5 · 硬件在环验证 / Hardware-in-the-loop validation

- Verdict: primary
- Support: 9 claims from 2 papers; typed chains: 0 (no_typed_chain; leaves 0, max depth 0)
- Exploratory (non-evidential) paths: 14 (20 nodes)
- Boundary: 1 · Oppose: 0
- Chain nodes: 0
- Statement: Before flight test, aircraft simulation design can be validated through real-time hardware-in-the-loop testing in which a ground computer solves the aircraft dynamics and the integrated control and propulsion systems are exercised under specified tracking-error criteria.

## Reading the numbers

- Coverage counts claims that are direct answer links *or* nodes on a **typed** support chain. A claim can appear in several roots when the routes overlap; overlap is reported, not hidden.
- Only `deduction` edges are chainable. `special_case`, `reproduces`, `bounds`, `abduction`, `analogy` and residual `infer` are shown but do not form support chains; `infer` with `residual_reason` is explicitly non-chainable.
- `exploratory_paths` are the old infer-based graph paths. They are useful for exploration but are **not counted** in direct support, typed chains, or coverage.
- `support` / `oppose` / `bounds` are LLM-judged links that were mechanically checked against the frozen candidate set. They are not a proof and not a vote count.
- This run starts from a claim-level graph without domain conclusions. In a full pipeline run with `summarize_conclusions: true`, the same layer can link answers to K conclusions first, then chains descend from K to claims.
- Root verdicts are structural: multi-paper support, single-paper support, opposed, or exploratory.
