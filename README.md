# vLLM推理服务压测工具

这是一个基于[vLLM](https://github.com/vllm-project/vllm)推理引擎的性能压测框架。该工具基于vLLM官方的[benchmarks](https://github.com/vllm-project/vllm/tree/main/benchmarks)目录提供的压测工具进行开发。

## 📌 功能特点

本工具用于评估vLLM在线推理服务的关键性能指标，包括：

* **延迟 (Latency)** - 请求响应时间
* **吞吐量 (Throughput)** - 每秒处理的请求数和token数
* **首token时间 (TTFT)** - 从发送请求到收到第一个token的时间
* **token间延迟 (ITL)** - 相邻token之间的生成间隔
* **每token输出时间 (TPOT)** - 平均每个输出token的生成时间
* **端到端延迟 (E2EL)** - 完整请求的总处理时间

该压测工具假设vLLM服务器运行在**OpenAI兼容**模式下。服务器设置说明请参考[vLLM快速开始指南](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)。

---

## 🔧 环境设置

使用以下命令安装所需的Python包：

```bash
pip install -r requirements.txt
```

### 依赖包说明

- `vllm` - vLLM推理引擎（用于分词器）
- `transformers` - HuggingFace变换器库
- `aiohttp` - 异步HTTP客户端
- `pandas` - 数据处理和分析
- `datasets` - HuggingFace数据集库

---

## ⚙️ 配置说明

编辑`config.yaml`文件来配置压测参数：

```yaml
# 模型和服务器配置
model: "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"        # 要压测的模型名称
base_url: "http://localhost:8010"                        # vLLM服务器URL
tokenizer: "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"   # 分词器路径

# 输入输出token长度组合
input_output:
  - [256, 256]    # [输入长度, 输出长度]
  - [512, 512]
  - [1024, 1024]

# 并发数和请求数组合  
concurrency_prompts:
  - [1, 10]       # [最大并发数, 总请求数]
  - [4, 40]
  - [8, 80]
  - [16, 160]
```

### 配置参数详解

- `model`: 在vLLM服务器中配置的模型名称
- `base_url`: vLLM服务器的基础URL地址
- `tokenizer`: 包含分词器文件的路径
- `input_output`: 输入和输出token长度的组合列表
- `concurrency_prompts`: 最大并发数和总请求数的组合列表

工具会对所有参数组合进行笛卡尔积测试，即每个输入输出长度组合都会在每个并发配置下进行测试。

---

## 🚀 运行压测

### 1. 启动vLLM服务器

首先确保vLLM服务器正在运行：

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --host 0.0.0.0 \
    --port 8010 \
    --swap-space 16 \
    --disable-log-requests
```

### 2. 执行批量压测

根据`config.yaml`中的配置开始压测：

```bash
python3 run.py
```

压测结果将保存在`results/`目录中，每个测试用例生成一个独立的`.json`文件。

### 3. 单独运行压测（可选）

你也可以直接使用`benchmark_serving.py`进行单次压测：

```bash
python3 benchmark_serving.py \
    --backend vllm \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --base-url http://localhost:8010 \
    --dataset-name random \
    --random-input-len 256 \
    --random-output-len 256 \
    --num-prompts 100 \
    --max-concurrency 10
```

---

## 📊 结果聚合

所有压测完成后，运行以下命令聚合结果：

```bash
python3 aggregate_result.py
```

这将生成一个汇总文件`results/aggregate_results.csv`，包含所有测试用例的性能指标。

### 输出指标说明

聚合结果包含以下关键指标：

- `completed`: 成功完成的请求数
- `request_throughput`: 请求吞吐量（请求/秒）
- `output_throughput`: 输出token吞吐量（token/秒）
- `mean_ttft_ms`: 平均首token时间（毫秒）
- `median_ttft_ms`: 首token时间中位数（毫秒）
- `p99_ttft_ms`: 首token时间99百分位数（毫秒）
- `mean_tpot_ms`: 平均每token输出时间（毫秒）
- `mean_itl_ms`: 平均token间延迟（毫秒）
- `mean_e2el_ms`: 平均端到端延迟（毫秒）

---

## 📁 项目结构

```
vllm_benchmark_serving/
├── backend_request_func.py      # 后端请求处理函数
├── benchmark_serving.py         # 主压测脚本
├── benchmark_dataset.py         # 数据集处理模块
├── benchmark_utils.py           # 工具函数
├── aggregate_result.py          # 结果聚合脚本
├── run.py                 # 批量执行脚本
├── config.yaml                  # 参数配置文件
├── requirements.txt             # Python依赖
├── README.md                    # 英文说明文档
├── README.md                 # 中文说明文档
└── results/                     # 压测结果目录
    ├── bench_io256x256_mc1_np10.json
    ├── bench_io256x256_mc4_np40.json
    └── aggregate_results.csv
```

---

## 🔍 代码结构说明

### 核心模块

1. **benchmark_serving.py** - 主压测脚本
   - 实现异步请求调度和性能指标收集
   - 支持多种数据集和后端
   - 提供详细的性能分析

2. **backend_request_func.py** - 后端请求处理
   - 实现与各种推理后端的通信
   - 支持OpenAI兼容API、TGI、DeepSpeed-MII等
   - 收集详细的延迟指标

3. **benchmark_dataset.py** - 数据集处理
   - 支持多种数据集格式（ShareGPT、Random、Sonnet等）
   - 提供统一的数据采样接口
   - 支持多模态数据处理

4. **run.py** - 批量执行控制器
   - 根据配置文件自动生成参数组合
   - 并行执行多个压测任务
   - 管理结果文件命名和存储

5. **aggregate_result.py** - 结果聚合工具
   - 合并多个JSON结果文件
   - 生成CSV格式的汇总报告
   - 提取关键性能指标

---

## 📈 使用建议

### 压测最佳实践

1. **预热测试**: 在正式压测前先运行少量请求预热模型
2. **逐步增压**: 从低并发开始，逐步增加并发数观察性能变化
3. **多次测试**: 每个配置运行多次取平均值，减少随机误差
4. **监控资源**: 压测时监控GPU/CPU/内存使用情况
5. **网络延迟**: 考虑客户端到服务器的网络延迟影响

### 参数调优建议

- **并发数**: 从1开始逐步增加，找到最佳吞吐量点
- **请求数**: 确保足够的样本量（建议至少100个请求）
- **输入长度**: 测试不同长度以评估模型在各种场景下的性能
- **输出长度**: 考虑实际应用场景的输出长度分布

---

## 🚨 注意事项

* 确保vLLM服务器在指定的`base_url`上运行并可访问
* 压测期间避免其他高负载任务影响结果准确性
* 根据实际需求调整`config.yaml`中的参数组合
* 大规模压测时注意服务器资源限制
* 建议在专用测试环境中运行压测，避免影响生产服务

---

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个工具！

如果你需要添加新的数据集支持、后端适配或性能指标，请参考现有代码结构进行扩展。

---

## 📄 许可证

本项目采用Apache-2.0许可证，详见各源文件头部的许可证声明。