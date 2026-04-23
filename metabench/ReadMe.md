# MetaBench 说明

这里保留的是 Meta-Writer 仓库内部使用的最小 benchmark 子模块。

它只负责三件事：

1. 保存本地 benchmark 样本；
2. 读取样本、输出与指标文件；
3. 计算七维分数并导出结果产物。

当前与主仓库的连接点是 `examples/benchmark_template.py`。

该接口会直接读取 `metabench/examples/samples.jsonl`，将样本转换为 Meta-Writer 可执行任务，并提供本地评估函数，便于在不依赖额外文档和外部 judge 的情况下完成联调。

最小可用目录如下：

- `metabench/examples/`：样本、示例输出、示例指标
- `metabench/config/baseline_cost.json`：成本归一化基线
- `metabench/src/metabench/`：读取、校验、打分与导出流水线

如需直接运行子模块，可在仓库根目录执行：

```bash
PYTHONPATH=metabench/src python -m metabench.cli \
  --samples metabench/examples/samples.jsonl \
  --outputs metabench/examples/outputs.jsonl \
  --metrics metabench/examples/metrics.jsonl \
  --baseline metabench/config/baseline_cost.json \
  --run-output-dir metabench/runs \
  --run-id demo_run_local \
  --model-name demo-model \
  --judge-model local-rule \
  --data-version v0.1 \
  --tokenizer-backend tiktoken \
  --tokenizer-name cl100k_base \
  --power-p -1
```

运行完成后会在 `metabench/runs/demo_run_local/` 生成 `samples.jsonl`、`summary.json` 与 `radar.csv`。
