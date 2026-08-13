# Figure 2 节点细化：Step 1–5 本地测试

处理对象：*One Ticket to Win Them All* 的 Figure 2（CIFAR-10a → CIFAR-10b 分布内迁移）。本文件依照 `pipeline.md`，为每一步保留一张“截至该步”的累计 Mermaid 图。

证据锚点：

- `01_paper_sample_one_ticket/paper_text_ocr_cleaned.md:98`：统一指标、6 个随机种子与 `mean ± 1 SD`；
- `01_paper_sample_one_ticket/paper_text_ocr_cleaned.md:102-115`：Figure 2、实验设置、比较臂、结果与作者解释；
- `01_paper_sample_one_ticket/images/867771291879342656_2.jpg`：两个 panel 的曲线与图例；
- `01_paper_sample_one_ticket/clean_claims_and_relations.json`：canonical assertion registry；
- `01_paper_sample_one_ticket/lkm_coarse_reasoning_graph.json`：C4、H5、P3 与粗推理步骤；
- `01_paper_sample_one_ticket/lth_coarse_graph_weakpoints.json`：`OTWTA-WC-07/08`。

Canonical assertion 复用：

- `A015`：CIFAR-10a 彩票迁移到互不重叠的 CIFAR-10b，并相对随机稀疏初始化改善训练；
- `A016`：该分布内迁移在 VGG19 和 ResNet50 上均成立；
- `A017`：ResNet50 在低剪枝比例下敏感，偶尔不如随机彩票。

标准化状态说明：仓库中没有发现 `pipeline.md` 所约定的可调用黑盒原子化/标准化工具。因此本文件不伪造 `standardized_new` 状态；`N01–N07` 均明确标为 `standardization_review`。它们可用于 Step 4/5 的本地逻辑草图，但在黑盒标准化、identify 和去重完成前，不是正式 Gaia-IR claim，也不能作为 Step 6 的 frozen registry 输入。

## Step 1：建立证据台账

本步只导入 assertion registry、定位来源并完成粗节点 identity。来源箭头、identity 箭头和 LKM 粗脉络均不是正式推理边。

```mermaid
flowchart TD
    CAP["来源：Figure 2 图注与图像<br/>CIFAR-10a/10b；VGG19 与 ResNet50；6 seeds"]:::src
    TXT["来源：§4.1 正文<br/>比较 10a ticket、10b ticket 与 random ticket"]:::src
    DISC["来源：§4.2 开头<br/>作者称结果表明 ticket 并未过拟合具体训练样本"]:::src

    C4["C4 LKM 粗结论<br/>分布内迁移 + 两架构范围 + ResNet50 低剪枝例外"]:::coarse
    H5["H5 LKM highlight<br/>留出子集迁移支持非样本记忆解释"]:::coarse
    P3["P3 LKM weak-point claim<br/>只测试 CIFAR-10 的一次 a/b 切分"]:::limit

    A15["A015 canonical assertion<br/>10a ticket 可迁移到不相交的 10b<br/>并相对随机初始化改善训练"]:::concl
    A16["A016 canonical assertion<br/>该迁移在 VGG19 与 ResNet50 上均成立"]:::concl
    A17["A017 canonical assertion<br/>ResNet50 低剪枝比例下偶尔不如 random"]:::caveat

    OVraw["O_V_raw Figure 2a 观察摘要<br/>VGG19：10a ticket 与 10b ticket 接近<br/>高剪枝时明显优于 random"]:::obs
    ORraw["O_R_raw Figure 2b 观察摘要<br/>ResNet50：极端剪枝时 tickets 优于 random<br/>低剪枝时反而低于 random"]:::obs
    Iraw["I_raw 作者解释摘要<br/>ticket 并未过拟合具体训练样本"]:::hyp

    CAP -. "原文/图像抽取" .-> OVraw
    CAP -. "原文/图像抽取" .-> ORraw
    TXT -. "实验比较抽取" .-> OVraw
    TXT -. "实验比较抽取" .-> ORraw
    DISC -. "作者解释抽取" .-> Iraw

    C4 -. "identify；非推理边" .-> A15
    C4 -. "identify；非推理边" .-> A16
    C4 -. "identify；非推理边" .-> A17
    OVraw -. "candidate evidence；非正式边" .-> A15
    ORraw -. "candidate evidence；非正式边" .-> A16
    ORraw -. "exact/semantic identify candidate" .-> A17
    H5 -. "粗解释；尚未形式化" .-> Iraw
    P3 -. "范围限制；尚未形式化" .-> C4

    classDef src fill:#f5f5f5,stroke:#757575
    classDef coarse fill:#eeeeee,stroke:#616161,stroke-dasharray:5 5
    classDef obs fill:#e8f5e9,stroke:#4caf50
    classDef hyp fill:#f3e5f5,stroke:#7b1fa2
    classDef concl fill:#e3f2fd,stroke:#1976d2
    classDef caveat fill:#fff3e0,stroke:#ef6c00
    classDef limit fill:#ffebee,stroke:#c62828
```

截至 Step 1：C4 已映射到 `A015/A016/A017`；H5 与 P3 保留为粗图来源。尚未补全实验字段，未确认推理类型。

## Step 2：补齐实验记录与结果 claim

Figure 2 只对应一个共享协议下的实验记录 `E_F2`。完整 S/A/B/M/R/U、两个 panel、曲线和 provenance 集中保存在 `E_F2` 中；`A015/A016/A017` 是从该记录 identify 出的三个已有原子结果命题，不是三个实验。

```mermaid
flowchart TD
    EF2["E_F2 Figure 2 单一实验记录（非 claim）<br/>S：CIFAR-10 分为不相交 10a/10b，各 25,000 张、每类 2,500 张；目标为 10b；测试 VGG19 与 ResNet50<br/>A：在 10a 发现并转移到 10b 训练的 winning ticket<br/>B：random sparse initialization；另以 10b-specific ticket 作参照<br/>M：收敛测试准确率随权重剪除比例变化<br/>R：10a ticket 可迁移并相对 random 改善训练；两架构均观察到迁移；ResNet50 低剪枝时存在反例<br/>U：6 seeds，mean ± 1 SD"]:::record

    A15["A015 canonical 主结果命题<br/>10a ticket 可迁移到不相交的 10b，<br/>并相对 random sparse initialization 改善训练"]:::claim
    A16["A016 canonical 架构范围命题<br/>A015 所述分布内迁移<br/>在 VGG19 与 ResNet50 上均成立"]:::claim
    A17["A017 canonical 条件性例外命题<br/>ResNet50 在低剪枝比例下较敏感，<br/>剪除较少权重时偶尔不如 random ticket"]:::caveat

    Iraw["I_raw 作者解释候选<br/>ticket 并未过拟合具体训练样本<br/>identity：无现成 assertion；standardization_review"]:::review
    P3["P3 范围限制<br/>单一数据集、一次确定性 a/b 切分<br/>未测试其他切分或数据集"]:::limit

    EF2 -. "主结果 identify/provenance；非推理边" .-> A15
    EF2 -. "架构范围 identify/provenance；非推理边" .-> A16
    EF2 -. "条件性例外 identify/provenance；非推理边" .-> A17
    A15 -. "核心结果；推理类型待 Step 3" .-> Iraw
    A16 -. "两架构范围；推理类型待 Step 3" .-> Iraw
    A17 -. "限制无条件读法；关系待 Step 3" .-> A16
    P3 -. "限制推广范围" .-> A15
    P3 -. "限制推广范围" .-> A16

    classDef record fill:#e8f5e9,stroke:#4caf50,stroke-dasharray:5 5
    classDef claim fill:#e3f2fd,stroke:#1976d2
    classDef caveat fill:#fff3e0,stroke:#ef6c00
    classDef limit fill:#ffebee,stroke:#c62828
    classDef review fill:#f3e5f5,stroke:#7b1fa2,stroke-dasharray:5 5
```

截至 Step 2：一个 `E_F2` 实验记录已经映射到三个已有原子结果命题；主结果、架构范围、条件性例外和作者解释相互分开。`E_F2` 不是 claim，实验操作与 identity/provenance 均不构成推理边；没有从图像人工估读精确数值或声称统计显著。

## Step 3：重新识别推理类型

- 从 CIFAR-10 的一次 a/b 切分及两个架构推广到“同分布不同样本通常可迁移”是 **induction**，不是 deduction；
- 从留出子集仍有收益推断“收益不依赖对具体源样本的记忆”是 **abduction / 排除性推理**；
- `A017` 对 `A015+A016` 的无条件读法构成低剪枝范围内的 **conditional contradiction/tension**；
- `E_F2` 只保存实验字段与 provenance，不参与这些推理。

```mermaid
flowchart TD
    EF2["E_F2 Figure 2 单一实验记录<br/>共享 S/A/B/M/R/U + 两个 panel；非 claim"]:::record
    A15["A015 canonical 主结果命题<br/>10a → 不相交 10b 的迁移收益"]:::claim
    A16["A016 canonical 架构范围命题<br/>分布内迁移在 VGG19 与 ResNet50 上均成立"]:::claim
    A17["A017 canonical 条件性例外命题<br/>ResNet50 低剪枝时偶尔低于 random"]:::caveat

    RIND(["候选 reasoning type<br/>induction<br/>一次切分、两个架构 → 有限范围的一般规律"]):::rtype
    Gsame["N01 候选一般规律<br/>在所测模型族中，winning ticket 的收益通常可在<br/>同一分布的不相交样本子集之间迁移<br/>standardization_review"]:::review

    RABD(["候选 reasoning type<br/>abduction / exclusion<br/>留出成功 → 非具体样本记忆解释"]):::rtype
    Hstruct["N02 候选作者解释<br/>ticket 携带可帮助同分布其他样本的初始化结构，<br/>收益不只依赖发现阶段见过的具体样本<br/>standardization_review"]:::review

    RX(["候选 relation<br/>conditional contradiction<br/>仅限 ResNet50 低剪枝范围"]):::rtype

    EF2 -. "主结果 identify；非推理边" .-> A15
    EF2 -. "架构范围 identify；非推理边" .-> A16
    EF2 -. "条件性例外 identify；非推理边" .-> A17
    A15 --> RIND
    A16 --> RIND
    RIND --> Gsame

    A15 --> RABD
    A16 --> RABD
    RABD --> Hstruct

    A16 --> RX
    A17 --> RX

    classDef record fill:#e8f5e9,stroke:#4caf50,stroke-dasharray:5 5
    classDef claim fill:#e3f2fd,stroke:#1976d2
    classDef caveat fill:#fff3e0,stroke:#ef6c00
    classDef rtype fill:#fff9c4,stroke:#f9a825,stroke-dasharray:5 5
    classDef review fill:#f3e5f5,stroke:#7b1fa2,stroke-dasharray:5 5
```

截至 Step 3：推理类型已重判，`E_F2` 仍只承担 provenance；连接仍是宏观摘要，尚未显式加入替代解释，也未用逻辑算子展开。

## Step 4：显式化 weakpoint

`WC-07` 与 `WC-08` 分开处理：前者质疑从单次切分向一般规律的推广，后者质疑“非样本记忆/可迁移结构”是否为留出成功的唯一解释。

```mermaid
flowchart TD
    EF2["E_F2 Figure 2 单一实验记录<br/>共享 S/A/B/M/R/U + 两个 panel；非 claim"]:::record
    A15["A015 canonical 主结果命题<br/>10a ticket 在不相交 10b 上有迁移收益"]:::claim
    A16["A016 canonical 架构范围命题<br/>该分布内迁移见于 VGG19 与 ResNet50"]:::claim
    A17["A017 canonical 条件性例外命题<br/>ResNet50 低剪枝时偶尔低于 random"]:::caveat

    W07(["OTWTA-WC-07<br/>induction weakpoint<br/>↝"]):::weak
    Gsame["N01 同分布样本间的一般迁移规律<br/>standardization_review"]:::review

    W08(["OTWTA-WC-08<br/>abduction / exclusion weakpoint<br/>↝"]):::weak
    Hstruct["N02 作者解释候选<br/>收益来自可帮助同分布其他样本的初始化结构，<br/>而非仅来自具体样本记忆<br/>standardization_review"]:::review

    P3["P3 范围限制<br/>只有 CIFAR-10 的一次确定性切分；<br/>没有其他 split 或 dataset 的独立重复"]:::limit
    AltSet["竞争解释候选（形式化人员补入）<br/>N03：该 a/b 切分存在偶然的样本选择或分布偏差<br/>N04：共同训练/剪枝协议产生配置特有效应<br/>N05：高稀疏度下的小样本正则化效应造成优势<br/>均为 standardization_review"]:::alt
    Hmem["N06 竞争假说<br/>ticket 收益只依赖发现时见过的具体 CIFAR-10a 样本<br/>standardization_review"]:::review

    EF2 -. "主结果 identify；非推理边" .-> A15
    EF2 -. "架构范围 identify；非推理边" .-> A16
    EF2 -. "条件性例外 identify；非推理边" .-> A17
    A15 --> W07
    A16 --> W07
    W07 --> Gsame

    A15 --> W08
    A16 --> W08
    W08 --> Hstruct

    P3 -. "限制 induction 范围" .-> W07
    AltSet -. "竞争解释；待 Step 5 展开" .-> W08
    Hmem -. "待由不相容预测检验" .-> W08
    A17 -. "低剪枝条件限制" .-> A16

    classDef record fill:#e8f5e9,stroke:#4caf50,stroke-dasharray:5 5
    classDef claim fill:#e3f2fd,stroke:#1976d2
    classDef caveat fill:#fff3e0,stroke:#ef6c00
    classDef weak fill:#fff9c4,stroke:#f9a825,stroke-dasharray:5 5
    classDef review fill:#f3e5f5,stroke:#7b1fa2,stroke-dasharray:5 5
    classDef limit fill:#ffebee,stroke:#c62828
    classDef alt fill:#fff3e0,stroke:#ef6c00,stroke-dasharray:5 5
```

截至 Step 4：`E_F2` 仍只承担实验记录与 provenance；`WC-07/08` 已诚实保留为 `↝`，P3、条件范围和竞争解释已经显式呈现，但尚未主观填写概率。

## Step 5：展开为细命题网络

展开决策：

- Figure 2 只保留一个非 claim 的 `E_F2` 实验记录；`A015/A016/A017` 是从该记录 identify 出的三个已有原子结果命题；
- `E_F2` 集中保存 S/A/B/M/R/U、两个 panel 与 provenance，但不参与正式推理；
- `A015` 同时承担 “Figure 2 待预测现象” 与 “实际 observation” 两个角色，`A016` 同时承担架构范围预测与观察角色；因为 canonical 内容相同，直接复用节点，不再制造重复的 `P₁/P₂`，也不画虚假的自等价边；
- `WC-07` 被两个可复用的单实例 induction/abduction 单元替换；
- `WC-08` 使用作者解释、替代解释、纯样本记忆假说及其不相容预测展开；
- `A017` 通过 `⊗` 保留对无条件架构范围读法的低剪枝限制；它不否定高剪枝区间的正结果；
- `N01–N07` 在黑盒标准化完成前保持 `standardization_review`，因此下图是 Step 5 逻辑草图，不是可冻结的正式 IR。

```mermaid
flowchart TD
    EF2["E_F2 Figure 2 单一实验记录（非 claim）<br/>S：不相交 CIFAR-10a/10b；目标 10b；VGG19 与 ResNet50<br/>A：10a transferred winning ticket<br/>B：random ticket；另以 10b-specific ticket 作参照<br/>M：收敛测试准确率随权重剪除比例变化<br/>R：主结果 + 两架构范围 + ResNet50 低剪枝例外<br/>U：6 seeds，mean ± 1 SD"]:::record

    Gsame["N01 候选一般规律<br/>在所测 VGG19/ResNet50 模型族中，winning ticket 的收益通常可在<br/>同一分布的不相交样本子集之间迁移<br/>standardization_review"]:::review

    AltSplit["N03 替代解释<br/>Figure 2 的正结果来自这一次 CIFAR-10a/b 切分的<br/>样本选择或偶然分布偏差<br/>standardization_review"]:::alt
    AltConfig["N04 替代解释<br/>两个 panel 的正结果来自共同训练/剪枝协议的<br/>配置特有效应，而非更一般的同分布规律<br/>standardization_review"]:::alt

    OR15{{"∨"}}
    OR16{{"∨"}}
    A15["A015 canonical 主结果命题<br/>10a ticket 迁移到不相交 10b，<br/>并相对 random 改善训练<br/>角色：预测现象 + 实际 observation"]:::claim
    A16["A016 canonical 架构范围命题<br/>该分布内迁移在 VGG19 与 ResNet50 上均成立<br/>角色：架构范围预测 + 实际 observation"]:::claim

    EF2 -. "主结果 identify/provenance；非推理边" .-> A15
    EF2 -. "架构范围 identify/provenance；非推理边" .-> A16
    Gsame --> OR15
    AltSplit --> OR15
    OR15 --> A15

    Gsame --> OR16
    AltConfig --> OR16
    OR16 --> A16

    Hstruct["N02 作者解释候选<br/>ticket 携带可帮助同分布其他样本的初始化结构，<br/>收益不只依赖发现阶段见过的具体样本<br/>standardization_review"]:::review
    AltReg["N05 竞争解释<br/>Figure 2 中出现迁移收益的剪枝区间可由<br/>小样本正则化效应解释，而不建立更一般的分布级结构机制<br/>standardization_review"]:::alt
    ORMECH{{"∨"}}

    Hstruct --> ORMECH
    AltSplit --> ORMECH
    AltReg --> ORMECH
    ORMECH --> A15

    Hmem["N06 纯样本记忆假说<br/>ticket 收益只依赖发现阶段见过的具体 CIFAR-10a 样本<br/>standardization_review"]:::review
    NoTransfer["N07 由纯样本记忆假说预测<br/>转移到不相交 CIFAR-10b 后不应优于 random<br/>standardization_review"]:::review
    XMEM{{"⊗"}}

    Hmem --> NoTransfer
    NoTransfer --- XMEM
    XMEM --- A15

    A17["A017 canonical 条件性例外命题<br/>ResNet50 在低剪枝比例下<br/>偶尔低于 random"]:::caveat
    XLOW{{"⊗<br/>low-pruning scope"}}
    EF2 -. "条件性例外 identify/provenance；非推理边" .-> A17
    A16 --- XLOW
    XLOW --- A17

    P3["P3 provenance/范围记录<br/>单一 CIFAR-10 a/b 切分；无其他切分或数据集复现"]:::limit
    P3 -. "使 N03 保持可行，限制 N01 的推广强度" .-> AltSplit

    classDef record fill:#e8f5e9,stroke:#4caf50,stroke-dasharray:5 5
    classDef claim fill:#e3f2fd,stroke:#1976d2
    classDef caveat fill:#fff3e0,stroke:#ef6c00
    classDef review fill:#f3e5f5,stroke:#7b1fa2,stroke-dasharray:5 5
    classDef alt fill:#fff3e0,stroke:#ef6c00,stroke-dasharray:5 5
    classDef limit fill:#ffebee,stroke:#c62828
```

截至 Step 5：`E_F2` 只承担实验记录与 provenance，A015/A016/A017 承担正式推理中的原子结果角色；`WC-07/08` 已被 `∨ / → / ⊗` 子网络替换。相同语义的 prediction/observation 已复用 A015/A016，没有为模板重复造 claim。A017 保留了 ResNet50 低剪枝区间的负结果，因此正结论只能作有条件解释。由于 `N01–N07` 仍处于 `standardization_review`，Step 5 结束冻结门禁尚未通过；下一步不是编码 Step 6，而是调用黑盒工具完成这些候选的原子化、identify、去重并按输出重画受影响的局部网络。
