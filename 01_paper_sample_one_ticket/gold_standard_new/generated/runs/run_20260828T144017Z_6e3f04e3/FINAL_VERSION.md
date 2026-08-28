# Pipeline v2.1 单篇论文最终版

该目录由用户于 2026-08-28 批准冻结。

- 最终实例：`run_20260828T144017Z_6e3f04e3`
- 输入：`gold_standard_new/paper_text.md` 与 `gold_standard_new/claims_final.json`
- Pipeline：`agent-pipeline-v2.1-step1-5`
- Pipeline 版本：`2.1.0`
- 最终状态：`succeeded`
- 结果：3 条 observation、3 条 Step 4 strategy、0 条未解决 weakpoint
- Gaia 官方实现：`gaia-lang 0.5.0a7`
- Viewer：`views/viewer.html`
- Pipeline 快照：`pipeline_version/`
  - Harness：`pipeline_version/agent/`
  - Pipeline v2.1：`pipeline_version/agent-pipeline-v2.1/`

## 冻结哈希

- Run：`e9f0a1dd2b4808e573a253ac560173611b4a11828302d83547c4cb3fa061cb32`
- 输入 manifest：`46b2b7487450b26cf86304d7972c7fc335ed41a138b5fa8b1ca0d59d7fb79fbb`
- Pipeline 配置：`031a65e87b02c5f20876372cb5ae860cc8bf27ff416f2e4da2a89bbcfc54c216`
- Viewer：`e6dd1a34ec5aa473405355f910184aeb09edf92370539ece18f2e79fdfcc9785`
- View model：`90c14ab2c42258c5e49f3fb2eba34e9b96be860935951903cd77ef6a4e60c907`
- Pipeline 快照树：`d779010eb9bba4d0a760cced4abf858771fc1490854003b7ce1383c7f68466f4`

首次 Step 2 attempt 因沙箱 DNS 限制失败，未获得模型响应；同一 run 的第二次 attempt 完成真实调用，之后 Step 3–5 全部成功。失败审计按原样保留。

`pipeline_version/` 是本次成功运行后立即保存的实际源码快照，仅排除 `.env`、`.DS_Store`、Python 缓存和 `*.pyc`，未修改 Pipeline 语义。
