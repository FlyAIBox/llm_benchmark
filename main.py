#!/usr/bin/env python3
# 文件: main.py
# 功能: vLLM推理服务压测工具统一入口脚本
# 说明: 集成批量压测、单次压测和结果聚合功能的完整工具

import sys
import os
import argparse
import yaml
import subprocess
import glob
import json
import pandas as pd
import re
from datetime import datetime

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# 常量定义
BENCHMARK_SCRIPT = "src/core/benchmark_serving.py"
DEFAULT_CONFIG_FILE = "config.yaml"
RESULT_DIR = "results"


def parse_input_output_lengths(filename):
    """
    从文件名中解析输入和输出token长度

    参数:
        filename (str): 结果文件名，格式如 "bench_io256x256_mc32_np128.json"

    返回:
        tuple: (input_len, output_len) 输入和输出token长度

    示例:
        "bench_io256x256_mc32_np128.json" → input_len=256, output_len=256
    """
    # 使用正则表达式匹配文件名中的io{数字}x{数字}模式
    match = re.search(r"io(\d+)x(\d+)", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def save_bilingual_csv(df, output_path):
    """
    保存双语CSV文件，第一行英文列名，第二行中文列名

    参数:
        df (DataFrame): 包含数据的DataFrame
        output_path (str): 输出文件路径
    """
    # 定义英文到中文的列名映射
    column_mapping = {
        'date': '日期',
        'backend': '后端',
        'model_id': '模型ID',
        'tokenizer_id': '分词器ID',
        'num_prompts': '提示数量',
        'request_rate': '请求速率',
        'burstiness': '突发性',
        'max_concurrency': '最大并发数',
        'duration': '持续时间',
        'completed': '完成数量',
        'total_input_tokens': '总输入令牌数',
        'total_output_tokens': '总输出令牌数',
        'request_throughput': '请求吞吐量',
        'request_goodput:': '请求有效吞吐量',
        'output_throughput': '输出吞吐量',
        'total_token_throughput': '总令牌吞吐量',
        'mean_ttft_ms': '平均首令牌时间(ms)',
        'median_ttft_ms': '中位首令牌时间(ms)',
        'std_ttft_ms': '首令牌时间标准差(ms)',
        'p99_ttft_ms': '首令牌时间P99(ms)',
        'mean_tpot_ms': '平均每令牌时间(ms)',
        'median_tpot_ms': '中位每令牌时间(ms)',
        'std_tpot_ms': '每令牌时间标准差(ms)',
        'p99_tpot_ms': '每令牌时间P99(ms)',
        'mean_itl_ms': '平均令牌间延迟(ms)',
        'median_itl_ms': '中位令牌间延迟(ms)',
        'std_itl_ms': '令牌间延迟标准差(ms)',
        'p99_itl_ms': '令牌间延迟P99(ms)',
        'mean_e2el_ms': '平均端到端延迟(ms)',
        'median_e2el_ms': '中位端到端延迟(ms)',
        'std_e2el_ms': '端到端延迟标准差(ms)',
        'p99_e2el_ms': '端到端延迟P99(ms)',
        'input_len': '输入长度',
        'output_len': '输出长度',
        'filename': '文件名'
    }

    # 获取中文列名
    chinese_columns = []
    for col in df.columns:
        chinese_columns.append(column_mapping.get(col, col))

    # 创建双语CSV内容
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        # 写入英文列名（第一行）
        f.write(','.join(df.columns) + '\n')
        # 写入中文列名（第二行）
        f.write(','.join(chinese_columns) + '\n')
        # 写入数据（从第三行开始）
        df.to_csv(f, index=False, header=False)


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
    os.makedirs(RESULT_DIR, exist_ok=True)

    # 生成结果文件名，格式: bench_io{输入长度}x{输出长度}_mc{并发数}_np{请求数}.json
    outfile = os.path.join(
        RESULT_DIR,
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


def batch_benchmark(config_file=DEFAULT_CONFIG_FILE):
    """
    批量执行压测实验

    参数:
        config_file (str): 配置文件路径

    处理流程:
    1. 从YAML配置文件加载参数设置
    2. 构建公共的命令行参数
    3. 遍历所有参数组合执行压测
    """
    # 从YAML配置文件加载设置和参数列表
    with open(config_file, "r") as f:
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


def single_benchmark(model, base_url, num_prompts=100, max_concurrency=10,
                    random_input_len=256, random_output_len=256, tokenizer=None):
    """
    执行单次压测

    参数:
        model (str): 模型名称
        base_url (str): vLLM服务器地址
        num_prompts (int): 请求数量
        max_concurrency (int): 最大并发数
        random_input_len (int): 输入长度
        random_output_len (int): 输出长度
        tokenizer (str): 分词器路径（可选）
    """
    # 构建命令行参数
    cmd = [
        'python3', BENCHMARK_SCRIPT,
        '--backend', 'vllm',
        '--model', model,
        '--base-url', base_url,
        '--dataset-name', 'random',
        '--random-input-len', str(random_input_len),
        '--random-output-len', str(random_output_len),
        '--num-prompts', str(num_prompts),
        '--max-concurrency', str(max_concurrency),
        '--percentile-metrics', 'ttft,tpot,itl,e2el'
    ]

    # 如果指定了分词器，添加到命令中
    if tokenizer:
        cmd.extend(['--tokenizer', tokenizer])

    # 创建结果保存目录
    os.makedirs(RESULT_DIR, exist_ok=True)

    # 生成结果文件名
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = os.path.join(RESULT_DIR, f"single_bench_{current_time}.json")
    cmd.extend(['--save-result', '--result-filename', outfile])

    # 打印即将执行的命令
    print(f"正在执行单次压测: {' '.join(cmd)}")

    # 执行压测命令
    ret = subprocess.run(cmd)

    # 检查执行结果
    if ret.returncode != 0:
        print(f"单次压测执行失败，返回码: {ret.returncode}")
    else:
        print(f"单次压测执行完成，结果已保存: {outfile}")


def aggregate_results():
    """
    聚合所有压测结果

    处理流程:
    1. 扫描results目录下的所有JSON文件
    2. 读取每个JSON文件的内容
    3. 从文件名解析输入输出长度信息
    4. 将所有数据合并到pandas DataFrame
    5. 导出为双语CSV文件（英文+中文列名）
    """
    # 1) 获取results目录下所有JSON文件的路径列表
    json_paths = glob.glob(os.path.join(RESULT_DIR, "*.json"))
    if not json_paths:
        print("在results/目录中未找到JSON文件。")
        return

    # 2) 遍历每个JSON文件，读取数据并添加元信息
    records = []
    for p in json_paths:
        # 读取JSON文件内容
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 获取文件名并解析输入输出长度
        filename = os.path.basename(p)
        input_len, output_len = parse_input_output_lengths(filename)

        # 向数据中添加额外的元信息
        data["input_len"] = input_len      # 输入token长度
        data["output_len"] = output_len    # 输出token长度
        data["filename"] = filename        # 原始文件名

        records.append(data)

    # 3) 使用pandas将所有记录转换为DataFrame
    df = pd.json_normalize(records)  # 将嵌套的JSON数据扁平化

    # 4) 生成带日期的输出CSV文件路径并保存为双语CSV文件
    current_date = datetime.now().strftime("%Y%m%d")
    out_csv = os.path.join(RESULT_DIR, f"aggregate_results_{current_date}.csv")
    save_bilingual_csv(df, out_csv)

    print(f"已聚合 {len(records)} 个测试结果 → {out_csv}")
    print("CSV文件格式：第一行为英文列名，第二行为中文列名，第三行开始为数据")


def main():
    """主函数：解析命令行参数并调用相应的功能"""
    parser = argparse.ArgumentParser(
        description="vLLM推理服务压测工具 - 集成批量压测、单次压测和结果聚合功能",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 批量压测（根据config.yaml配置）
  python main.py batch
  python main.py batch --config custom_config.yaml

  # 单次压测
  python main.py single --model deepseek-ai/DeepSeek-R1 --base-url http://localhost:8010 --num-prompts 100
  python main.py single --model /path/to/model --base-url http://localhost:8010 --max-concurrency 16 --random-input-len 512 --random-output-len 512

  # 聚合结果
  python main.py aggregate

功能说明:
  batch     - 根据config.yaml配置文件执行批量压测，支持多种参数组合的笛卡尔积测试
  single    - 执行单次压测，适合快速测试特定参数配置
  aggregate - 聚合results目录下的所有JSON结果文件，生成双语CSV报告
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 批量压测子命令
    batch_parser = subparsers.add_parser('batch', help='批量压测（根据config.yaml配置）')
    batch_parser.add_argument('--config', default=DEFAULT_CONFIG_FILE,
                             help=f'配置文件路径 (默认: {DEFAULT_CONFIG_FILE})')

    # 单次压测子命令
    single_parser = subparsers.add_parser('single', help='单次压测')
    single_parser.add_argument('--model', required=True, help='模型名称或路径')
    single_parser.add_argument('--base-url', required=True, help='vLLM服务器地址 (例如: http://localhost:8010)')
    single_parser.add_argument('--num-prompts', type=int, default=100, help='请求数量 (默认: 100)')
    single_parser.add_argument('--max-concurrency', type=int, default=10, help='最大并发数 (默认: 10)')
    single_parser.add_argument('--random-input-len', type=int, default=256, help='输入token长度 (默认: 256)')
    single_parser.add_argument('--random-output-len', type=int, default=256, help='输出token长度 (默认: 256)')
    single_parser.add_argument('--tokenizer', help='分词器路径 (可选，默认使用模型路径)')

    # 聚合结果子命令
    agg_parser = subparsers.add_parser('aggregate', help='聚合压测结果')

    args = parser.parse_args()

    if args.command == 'batch':
        print("=== 开始批量压测 ===")
        print(f"使用配置文件: {args.config}")
        batch_benchmark(args.config)
        print("=== 批量压测完成 ===")

    elif args.command == 'single':
        print("=== 开始单次压测 ===")
        print(f"模型: {args.model}")
        print(f"服务器地址: {args.base_url}")
        print(f"请求数量: {args.num_prompts}")
        print(f"最大并发数: {args.max_concurrency}")
        print(f"输入长度: {args.random_input_len}")
        print(f"输出长度: {args.random_output_len}")

        single_benchmark(
            model=args.model,
            base_url=args.base_url,
            num_prompts=args.num_prompts,
            max_concurrency=args.max_concurrency,
            random_input_len=args.random_input_len,
            random_output_len=args.random_output_len,
            tokenizer=args.tokenizer
        )
        print("=== 单次压测完成 ===")

    elif args.command == 'aggregate':
        print("=== 开始聚合结果 ===")
        aggregate_results()
        print("=== 结果聚合完成 ===")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()