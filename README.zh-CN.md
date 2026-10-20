# Gaia IR

[English](README.md) · [简体中文](README.zh-CN.md)

将论文主张编译为 Gaia IR，再把多个论文包合并为跨论文推理图。本仓库保留运行框架、提取流程、流水线配置及测试。需要按问题检索论文并端到端运行时，请使用 [Ariadne](https://github.com/CatherineWuleyan/ariadne)。

## 安装

需要 Python 3.11 或更高版本，编译器依赖固定为 `gaia-lang==0.5.0a7`。

```bash
git clone https://github.com/CatherineWuleyan/gaia-ir.git
cd gaia-ir
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e agent

export PYTHONPATH="$PWD/agent:$PWD/pipelines/single_paper:$PWD/pipelines/merge"
```

在线提取和推理需要模型密钥。已测试的客户端使用兼容 OpenAI 的接口：

```bash
export DEEPSEEK_API_KEY="your-key"
export DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"
export DEEPSEEK_MODEL="deepseek-v4-flash"
```

## 运行

正式配置为 `pipelines/single_paper/pipeline.json`、`pipelines/merge/pipeline.json` 和 `pipelines/merge/pipeline.question.json`。仅用于测试单个阶段的配置放在测试夹具目录中。

### 单篇论文

准备输入清单，路径相对清单文件解析：

```json
{
  "artifacts": [
    {"path": "paper.md", "kind": "source.paper_text", "media_type": "text/markdown"},
    {"path": "claims_final.json", "kind": "source.claims_final", "media_type": "application/json"}
  ]
}
```

```bash
./run_pipeline.sh single-v2 /absolute/path/to/inputs.json
```

如需先从论文图提取主张，将图放在 `pipelines/clean_claims/data/<paper-id>/graph.json`，然后运行：

```bash
python pipelines/clean_claims/run_full_pipeline.py <paper-id>
```

### 合并论文

先导出每个已完成的单篇运行：

```bash
python pipelines/single_paper/export_artifacts.py /path/to/run paper-id --out /path/to/packages
```

导出器在 `graph_single/<paper-id>/` 下写入 `gaia.ir.json`、`formalization.json` 和 `knowledge.index.json`。为每篇论文将这三个文件列入合并输入清单，产物类型分别为 `gaia.ir`、`formalization` 和 `knowledge.index`。

在 `pipelines/merge/pipeline.json` 第一阶段中设置 `scope.domain` 和 `scope.selection`，选中项必须与各论文包的 `namespace:name` 一致。然后运行：

```bash
./run_pipeline.sh merge /absolute/path/to/merge-inputs.json
```

若需要问题回答层，在 `pipeline.question.json` 中配置论文范围和问题，再通过 `pipeline_harness init --pipeline` 加载该配置。

### 检查或恢复运行

```bash
python -m pipeline_harness status --run /path/to/run
python -m pipeline_harness run --run /path/to/run
python -m pipeline_harness view --run /path/to/run
python -m pipeline_harness check --run /path/to/run
```

用浏览器打开生成的 `views/viewer.html`。生成的运行目录和本地输入不会提交到 Git。

## 离线检查

内置的合成数据流程可以在不调用模型的情况下检查运行框架：

```bash
run_dir=$(python -m pipeline_harness init --pipeline synthetic --runs-root outputs/smoke)
python -m pipeline_harness run --run "$run_dir"
python -m pipeline_harness view --run "$run_dir"
python -m pipeline_harness check --run "$run_dir"
```

## 测试

按上述方式激活环境并设置 `PYTHONPATH` 后运行：

```bash
python -m unittest discover -s agent/tests -p 'test_*.py'
python -m unittest discover -s pipelines/single_paper/tests -p 'test_*.py'
python -m unittest discover -s pipelines/merge/tests -p 'test_*.py'
```

## 目录

```text
agent/                         运行框架、产物存储、编译器集成、交互图和测试
pipelines/clean_claims/         论文图 → 提取主张
pipelines/single_paper/         主张 → Gaia IR；论文包导出器
pipelines/merge/                跨论文合并与可选回答层
requirements.txt               运行依赖
run_pipeline.sh                单篇与合并入口
```
