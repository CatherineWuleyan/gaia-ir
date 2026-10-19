# Ariadne 0.4.1 快速开始

Ariadne 是问题驱动的知识图谱研究运行时：输入一个领域问题，自动检索 LKM 论文，
构建或读取论文图，跑 paper2ir / single / merge，最后生成候选答案 root、类型化
推理链和自包含问题图。

这个压缩包已经包含 Ariadne、所需 pipeline、离线飞机示例和安装脚本。

## 环境要求

- macOS / Linux（Windows 用 WSL）
- Python **3.11+**
- 可访问模型 API 和 Bohr LKM 的网络
- OpenAI-compatible 模型 API key（DeepSeek 是实际测试路径）
- Bohr 账户（跑离线 `demo` 不需要）

## 1. 安装

```bash
tar -xzf ariadne-0.4.1-bundle.tar.gz
cd ariadne-0.4.1
./install.sh
```

安装脚本会创建 `./.venv`，安装 `ariadne/requirements.txt` 和内置 pipeline
harness。已有 Python 环境可以用：

```bash
./install.sh --no-venv
```

## 2. 配置自己的 API key

```bash
./ariadne/run.sh config init
```

按提示输入模型 API key、base URL、模型名，以及可选的 Bohr access key。
key 保存在 `~/.config/ariadne/config.json`，权限 `0600`，不会打进安装包。

也可以用环境变量：

```bash
export DEEPSEEK_API_KEY="sk-..."
export DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"
export DEEPSEEK_MODEL="deepseek-v4-flash"
export BOHR_ACCESS_KEY="..."
```

检查：

```bash
./ariadne/run.sh doctor
./ariadne/run.sh config test --api
./ariadne/run.sh config test --bohr
```

## 3. 离线 demo（不需要 key）

```bash
./ariadne/run.sh demo
open outputs/ariadne_runs/run_*/domain_viewer_question.html
```

## 4. 跑真实问题

```bash
./ariadne/run.sh ask "怎么完成飞机的仿真设计？" --top-k 2 --yes
```

输出在 `outputs/ariadne_runs/`：

- `domain_viewer_question.html`：问题图；
- `ALL_CHAINS_TO_ROOTS.md`：全部多步链；
- `05_outputs/answer_layer.json`：候选答案 root 和 answer links；
- `01_retrieval/selection.json`：每条查询、候选、图探测、跳过原因和最终选文；
- `01_retrieval/extract/`：用 `bohr lkm extract` 补图时的审计。

第一次跑建议：

```bash
./ariadne/run.sh ask "你的问题" --top-k 2 --no-fulltext --yes
```

用 `--dry-run` 可以只看计划，不花钱。

## 包内含什么

```text
ariadne/                 Ariadne 启动器、CLI、检索、paper2ir、render、测试
agent/                   single/merge 使用的 pipeline harness
pipelines/clean_claims/  paper2ir
pipelines/single_paper/  单篇 pipeline
pipelines/merge/         领域图 + 问题/答案层
weekly_reports/          single artifact exporter
outputs/aircraft/final/  冻结的飞机 demo 图
```

## 包内没有什么

- API key、Bohr 凭据、`.env`
- 本地论文语料和生成的 `outputs/ariadne_runs/`
- 历史运行、私有数据集和完整研究工作区

## 文档

- [中文完整说明](ariadne/README.zh-CN.md)
- [English guide](ariadne/README.md)
- [配置示例](ariadne/config.example.json)

## 常见问题

- **Python 版本不够**：安装 3.11+，然后
  `./install.sh --python /path/to/python3`。
- **`doctor` 说缺少 API key**：跑 `config init` 或设置
  `DEEPSEEK_API_KEY`。
- **Bohr 没登录**：`./ariadne/run.sh auth bohr`。
- **完整运行很慢**：paper2ir 每篇 14 步，先用 `--top-k 2`。
- **论文没有原生图**：Ariadne 会尝试 `bohr lkm extract`（成功 1.00 CNY/篇，
  受 `--max-extractions` 限制）；`--no-lkm-extract` 可关闭。
