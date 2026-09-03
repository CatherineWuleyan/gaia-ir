# Historical pipeline archive

本目录是可恢复的历史封存区，不是当前运行入口。

- `runs/round1`–`runs/round4` 保存历史运行产物、配置快照、checkpoint 和审计结果；其中的 `pipeline.commit.txt` 记录当时使用的源码提交。
- `legacy-entrypoints/` 保存曾固定指向 `/workspace/gaia-round2` 或 `/workspace/gaia-round3` 的 Bohr 作业脚本。
- 当前本地运行只允许使用仓库根目录的 `./run_pipeline.sh`，并选择 `single-v2`、`merge` 或 `legacy`。

封存内容可用于复现和诊断，但不得被当作当前源码或默认 pipeline。删除前应先确认不再需要对应 round 的审计证据。
