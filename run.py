# 文件: run.py
# 功能: vLLM推理压测批量执行脚本
# 说明: 根据配置文件中的参数组合，自动执行多组压测实验
# 
# 目录结构:
# ├── config.yaml      # YAML格式的参数组合配置文件
# └── run.py     # 批量执行压测的驱动脚本
#
# 依赖: pip install pyyaml

import yaml
import subprocess
import os

# benchmark_serving.py脚本的路径（根据实际位置调整相对路径）
BENCHMARK_SCRIPT = "benchmark_serving.py"
# YAML配置文件路径
CONFIG_FILE = "config.yaml"


def run_benchmark(common_args, input_len, output_len, concurrency, num_prompts):
    """
    执行单个参数组合的压测
    
    参数:
        common_args (list): 公共的命令行参数列表
        input_len (int): 输入token长度
        output_len (int): 输出token长度  
        concurrency (int): 最大并发请求数
        num_prompts (int): 总请求数量
    """
    # 复制公共参数，避免修改原始列表
    args = common_args.copy()
    
    # 添加输入和输出token长度参数
    args += ["--random-input-len", str(input_len), "--random-output-len", str(output_len)]
    
    # 添加并发控制和请求数量参数
    args += ["--max-concurrency", str(concurrency)]
    args += ["--num-prompts", str(num_prompts)]

    # 创建结果保存目录
    result_dir = "results"
    os.makedirs(result_dir, exist_ok=True)
    
    # 生成结果文件名，格式: bench_io{输入长度}x{输出长度}_mc{并发数}_np{请求数}.json
    outfile = os.path.join(
        result_dir,
        f"bench_io{input_len}x{output_len}_mc{concurrency}_np{num_prompts}.json"
    )
    args += ["--save-result", "--result-filename", outfile]

    # 打印即将执行的命令
    print(f"正在执行: {' '.join(args)}")
    
    # 执行压测命令
    ret = subprocess.run(args, capture_output=True, text=True)
    
    # 检查执行结果
    if ret.returncode != 0:
        print(f"参数组合 io=({input_len},{output_len}), mc={concurrency}, np={num_prompts} 执行失败: {ret.stderr}")
    else:
        print(f"参数组合 io=({input_len},{output_len}), mc={concurrency}, np={num_prompts} 执行完成，结果已保存: {outfile}")


def main():
    """
    主函数：批量执行压测实验
    
    处理流程:
    1. 从YAML配置文件加载参数设置
    2. 构建公共的命令行参数
    3. 遍历所有参数组合执行压测
    """
    # 从YAML配置文件加载设置和参数列表
    with open(CONFIG_FILE, "r") as f:
        cfg = yaml.safe_load(f)
    
    # 提取配置参数
    model = cfg["model"]                              # 模型名称
    base_url = cfg["base_url"]                        # vLLM服务器地址
    tokenizer = cfg["tokenizer"]                      # 分词器路径
    io_pairs = cfg.get("input_output", [])            # 输入输出长度组合列表
    cp_pairs = cfg.get("concurrency_prompts", [])     # 并发数和请求数组合列表

    # 构建所有压测共用的命令行参数
    common_args = [
        "python3", BENCHMARK_SCRIPT,
        "--backend", "vllm",                          # 使用vLLM后端
        "--model", model,                             # 指定模型
        "--base-url", base_url,                       # vLLM服务器URL
        "--tokenizer", tokenizer,                     # 分词器
        "--dataset-name", "random",                   # 使用随机生成的数据集
        "--percentile-metrics", "ttft,tpot,itl,e2el"  # 要统计的性能指标百分位数
    ]

    # 执行所有参数组合的压测（笛卡尔积）
    # 对每个输入输出长度组合，测试所有的并发数和请求数组合
    for input_len, output_len in io_pairs:
        for concurrency, num_prompts in cp_pairs:
            run_benchmark(common_args, input_len, output_len, concurrency, num_prompts)


if __name__ == "__main__":
    main()
