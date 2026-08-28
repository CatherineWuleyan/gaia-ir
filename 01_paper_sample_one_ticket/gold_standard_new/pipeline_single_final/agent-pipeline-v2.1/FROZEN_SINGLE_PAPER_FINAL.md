# Pipeline v2.1 单篇论文最终冻结版本

冻结日期：2026-08-28

状态：**FINAL / FROZEN**

本冻结只覆盖当前的单篇论文 Pipeline v2.1（Step 1–5）。后续任何代码、Prompt、Pipeline 配置、Gaia 版本或输入变化，都必须生成新的版本与新的完整验收 run；不得把旧 run 重新标记为本冻结版本。

## 冻结范围

- Pipeline：`pipeline.step1-5.json`，版本 `2.1.0`
- Pipeline 配置 SHA-256：`031a65e87b02c5f20876372cb5ae860cc8bf27ff416f2e4da2a89bbcfc54c216`
- Harness 冻结配置 SHA-256：`40733ff7e8009ef69edad5651cf3d3eda9898ffe701dc8d3eba1c8a267b10879`
- 官方 Gaia：`gaia-lang==0.5.0a7`
- 官方 IR schema：`ir-v1+ec83f1ad757a`
- Step 2 实现 SHA-256：`383a5ca161f9dc9937bb35a289c6b0ce75a2339c7419606354e4950652dc9c86`
- Gaia compiler adapter SHA-256：`e0ed55f87ef21b25faba8d94faff09283e114324fcbdea0f8cf45622a013889f`
- Harness 依赖配置 SHA-256：`2efcfd7d855d9727e92df38895934eb0da6b0010a84adaac0c5dc7262eedfe89`

## 冻结输入

- `gd_new` claims：`../01_paper_sample_one_ticket/2.1gd_new/claims_final.json`
  - SHA-256：`cbd31a11086190f24edaca85829e6003877d2193eabbba2596b9f2a371eedfce`
- 论文全文：`../01_paper_sample_one_ticket/pipeline_v2_iterations/round_12_parallel_step4/inputs/full_article/paper_text.md`
  - SHA-256：`ef309a34b05f8bf43d6e074fd6fa6cef9287f0284e59c7318bef933e7675ac96`

`../01_paper_sample_one_ticket/2.1gd_new/paper_text.md` 是 15 行局部切片，不属于本冻结输入。

## 最终验收 run

- Run：`../01_paper_sample_one_ticket/2.1gd_new/generated/final-validation-a7/run_20260828T123651Z_92cf6d6c`
- 状态：`succeeded`
- Findings：0
- 六个阶段均为第一次 attempt 成功
- 最终 checkpoint：`checkpoint_3769c306b8394f43b96aa03dc8bd45a1`

### 正式产物

- Step 4 formalization：`artifact_4201ffd9cf964740961599d82be97feb`
  - Artifact SHA-256：`9639d1cac9f2ccd19a6c692d0e36864413cf7c323a03861214d25f82f30d793a`
  - Revision content hash：`0f4fe068a99864ffa9c325a40219f70104522d81ae66b1ba8d4484161c24504b`
- 官方 Gaia IR：`artifact_1f69e55c3cf0409791451247d68b4b65`
  - Artifact SHA-256：`dc44636d71ce5682467ac216120c1290aa73b5011e7bcb9baa0dcae94f314dd5`
  - IR hash：`sha256:6e975c7ec7661e32cb7c27253b9436d522ec130bc89ef330fbdf00a12d272507`
- Knowledge index：`artifact_68f5532d714249d69e0e410ef55604a4`
  - Artifact SHA-256：`869108d9df0ae74aa771d860d462040d0de3dfc0e714de75f667ba45b95a3af6`
- 官方编译审计：`artifact_f0e4a5ccbea244098045fd4e01c0326d`
  - Tool：`gaia-0.5-official-compiler` `0.5.0a7`
  - Contract status：`official`

## 验收结论

- v2.1 测试：67/67 通过
- 共享 Harness 测试：43/43 通过
- Step 1–4 formalization validation：全部 `passed`，0 error、0 warning、0 finding
- `pipeline_harness check`：`ok: true`，0 finding
- Step 2 对 Figure 1 的前七个窗口均返回 `insufficient_context`，第八个窗口纳入本文结果段后才抽取；没有把 related work 当作本文实验
- Step 3 从单次冻结的 Step 2 revision 生成 weakpoint，Step 4 从单次冻结的 Step 3 revision 并行只读处理并最终一次合并
- Step 4 最终包含 33 个 knowledge、15 个 operator、2 个 strategy、0 个未处理 weakpoint
- AltExp 由官方 Gaia 编译派生；Viewer 仅以 `alternative_placeholder` 在 `helpers` 层展示
- Viewer search 中没有 helper 或 AltExp 占位符；helper 不作为普通公开 claim 展示

## 非最终 run

`../01_paper_sample_one_ticket/2.1gd_new/generated/final-validation/run_20260828T090620Z_ac57ac9b` 使用 `gaia-lang 0.5.0a5`，仅保留为升级审计记录，不是最终冻结版本。
