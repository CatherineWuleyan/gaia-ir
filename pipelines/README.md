# Pipeline entry points

仓库只把下面三条流程作为可直接启动的顶层入口。统一从仓库根目录执行
`./run_pipeline.sh`，启动器会打印 pipeline 名称、配置绝对路径、配置 hash、
Git commit、工作树是否干净以及实际导入的 Python 模块路径。

| 名称 | 配置/脚本 | 作用 |
|---|---|---|
| `single-v2` | `pipelines/single_paper/pipeline.step1-5.json` | 新的单论文 Harness V2，完整 Step 1–5 |
| `merge` | `pipelines/merge/pipeline.step5.json` | 多论文 merge，Step 0–5 |
| `legacy` | `pipelines/single_paper/agent_pipeline_v2/claim_cleaner/run_full_pipeline.py` | 原始 14 步清洗流程 |

示例：

```bash
./run_pipeline.sh list
./run_pipeline.sh single-v2 /absolute/path/to/manifest.json
./run_pipeline.sh merge /absolute/path/to/manifest.json
./run_pipeline.sh legacy 1032903864883347458
```

`pipeline.step1.json`、`pipeline.step2.json`、`pipeline.step3.json` 和
`pipeline.step4.json` 是局部开发/测试配置，不是“从完整 run 的对应步骤继续”的
快捷入口。直接对它们执行 `init` 会创建一个只有局部 stage 的新 run；需要继续
历史 run 时，应对完整 pipeline 的成功 checkpoint 使用 Harness 的 `fork`。

不要使用 `bohr_round2_runner.sh` 或 `bohr_round4_recover.sh` 作为本地入口：它们
固定指向 `/workspace/gaia-round2`/`/workspace/gaia-round3` 的历史环境，适合只在
对应 Bohr 作业环境中复现旧 round，不代表当前仓库源码。
