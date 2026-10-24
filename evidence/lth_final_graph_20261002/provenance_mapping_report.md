# merge.html provenance audit

该报告只使用最终 `merge.html`、`outputs/merge_clusters/cluster_*/view_model.json` 和本地 `data/*` 文件。
生成的候选 claim 通过 cluster 推理边反向追踪到论文 claim；没有路径的节点和边保留为 unknown。

## 统计

- 图节点：240；direct：0；derived：220；unresolved：20。
- 图边：235；direct：0；derived：125；unresolved：110。
- 源 claim：467；源论文：79；有原文行号的 paragraph anchor：1342 次（唯一 anchor 238）；claim-file anchor：282 次。

## 解释

`resolved_derived` 表示图中的汇总 claim 由 cluster 推理边指向的一组源 claim 支持；它不是源论文的逐字引用。
`anchor_claim_N` 只证明 claim-file 中的第 N 条 claim，不等于原文位置。只有 `anchor_paragraph_pXXXX` 且能在 `ocr.md` 中解析出行号时，才计入原文 anchor。
最终 merge 边没有保留单独的 source edge anchor，因此边只报告两端节点的 source-claim closure，并明确标记为 derived。

## unknown 节点

- `knowledge:candidate_A_1f3724500e342844` (A117)
- `knowledge:candidate_A_47b862588d2a3466` (A118)
- `knowledge:candidate_A_97fd0f1a73e1e952` (A119)
- `knowledge:candidate_A_a5eada4915035f25` (A120)
- `knowledge:candidate_A_d9aaa1f3f1eab205` (A121)
- `knowledge:candidate_A_de95eb24af643ff8` (A122)
- `knowledge:candidate_A_fdd13dabebf16646` (A123)
- `knowledge:ref:K01` (K01)
- `knowledge:ref:K02` (K02)
- `knowledge:ref:K03` (K03)
- `knowledge:ref:K04` (K04)
- `knowledge:ref:K05` (K05)
- `knowledge:ref:K06` (K06)
- `knowledge:ref:K07` (K07)
- `knowledge:ref:K08` (K08)
- `knowledge:ref:K09` (K09)
- `knowledge:ref:K10` (K10)
- `knowledge:ref:K11` (K11)
- `knowledge:ref:K12` (K12)
- `knowledge:ref:K13` (K13)
