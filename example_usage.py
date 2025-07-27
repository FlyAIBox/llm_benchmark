#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
vLLM推理服务压测工具使用示例

本脚本演示了如何使用vLLM压测工具进行性能测试的完整流程。
包括配置检查、单次测试、批量测试和结果分析等功能。

使用方法:
    python3 example_usage.py
"""

import os
import sys
import json
import yaml
import subprocess
import time
from pathlib import Path

def check_vllm_server(base_url: str) -> bool:
    """
    检查vLLM服务器是否正在运行
    
    参数:
        base_url: vLLM服务器地址
        
    返回:
        bool: 服务器是否可访问
    """
    import aiohttp
    import asyncio
    
    async def check_server():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{base_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                    return response.status == 200
        except:
            return False
    
    return asyncio.run(check_server())

def load_config(config_file: str = "config.yaml") -> dict:
    """
    加载配置文件
    
    参数:
        config_file: 配置文件路径
        
    返回:
        dict: 配置字典
    """
    if not os.path.exists(config_file):
        print(f"❌ 配置文件 {config_file} 不存在")
        return None
        
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✅ 成功加载配置文件: {config_file}")
        return config
    except Exception as e:
        print(f"❌ 加载配置文件失败: {e}")
        return None

def run_single_test(config: dict) -> bool:
    """
    运行单次测试以验证配置
    
    参数:
        config: 配置字典
        
    返回:
        bool: 测试是否成功
    """
    print("\n🧪 运行单次测试验证配置...")
    
    # 构建测试命令
    cmd = [
        "python3", "benchmark_serving.py",
        "--backend", "vllm",
        "--model", config["model"],
        "--base-url", config["base_url"],
        "--tokenizer", config["tokenizer"],
        "--dataset-name", "random",
        "--random-input-len", "128",
        "--random-output-len", "128",
        "--num-prompts", "5",
        "--max-concurrency", "1",
        "--disable-tqdm"
    ]
    
    try:
        print(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ 单次测试成功完成")
            return True
        else:
            print(f"❌ 单次测试失败: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ 单次测试超时（5分钟）")
        return False
    except Exception as e:
        print(f"❌ 单次测试异常: {e}")
        return False

def run_batch_tests() -> bool:
    """
    运行批量测试
    
    返回:
        bool: 批量测试是否成功
    """
    print("\n🚀 开始批量压测...")
    
    try:
        result = subprocess.run(["python3", "run.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 批量压测完成")
            return True
        else:
            print(f"❌ 批量压测失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 批量压测异常: {e}")
        return False

def aggregate_results() -> bool:
    """
    聚合测试结果
    
    返回:
        bool: 结果聚合是否成功
    """
    print("\n📊 聚合测试结果...")
    
    try:
        result = subprocess.run(["python3", "aggregate_result.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 结果聚合完成")
            return True
        else:
            print(f"❌ 结果聚合失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 结果聚合异常: {e}")
        return False

def show_results():
    """
    显示测试结果摘要
    """
    print("\n📈 测试结果摘要:")
    
    # 检查结果文件
    results_dir = Path("results")
    if not results_dir.exists():
        print("❌ 结果目录不存在")
        return
    
    # 统计JSON结果文件
    json_files = list(results_dir.glob("*.json"))
    print(f"📁 生成的结果文件数量: {len(json_files)}")
    
    # 显示CSV聚合结果
    csv_file = results_dir / "aggregate_results.csv"
    if csv_file.exists():
        print(f"📄 聚合结果文件: {csv_file}")
        
        try:
            import pandas as pd
            df = pd.read_csv(csv_file)
            print(f"📊 测试用例数量: {len(df)}")
            
            # 显示关键指标摘要
            if 'request_throughput' in df.columns:
                print(f"🚀 平均请求吞吐量: {df['request_throughput'].mean():.2f} req/s")
            if 'mean_ttft_ms' in df.columns:
                print(f"⚡ 平均首token时间: {df['mean_ttft_ms'].mean():.2f} ms")
            if 'mean_tpot_ms' in df.columns:
                print(f"🔄 平均每token时间: {df['mean_tpot_ms'].mean():.2f} ms")
                
        except Exception as e:
            print(f"❌ 读取CSV文件失败: {e}")
    else:
        print("❌ 聚合结果文件不存在")

def main():
    """
    主函数：执行完整的压测流程
    """
    print("=" * 60)
    print("🎯 vLLM推理服务压测工具 - 使用示例")
    print("=" * 60)
    
    # 1. 加载配置
    config = load_config()
    if not config:
        sys.exit(1)
    
    print(f"📋 配置信息:")
    print(f"   模型: {config['model']}")
    print(f"   服务器: {config['base_url']}")
    print(f"   输入输出组合: {config['input_output']}")
    print(f"   并发组合: {config['concurrency_prompts']}")
    
    # 2. 检查服务器状态
    print(f"\n🔍 检查vLLM服务器状态...")
    if not check_vllm_server(config['base_url']):
        print(f"❌ 无法连接到vLLM服务器: {config['base_url']}")
        print("请确保vLLM服务器正在运行，例如:")
        print(f"vllm serve {config['model']} --host 0.0.0.0 --port 8010")
        sys.exit(1)
    
    print("✅ vLLM服务器连接正常")
    
    # 3. 运行单次测试
    if not run_single_test(config):
        print("❌ 单次测试失败，请检查配置")
        sys.exit(1)
    
    # 4. 询问是否继续批量测试
    response = input("\n❓ 单次测试成功，是否继续批量压测？(y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("👋 测试结束")
        return
    
    # 5. 运行批量测试
    start_time = time.time()
    if not run_batch_tests():
        print("❌ 批量测试失败")
        sys.exit(1)
    
    # 6. 聚合结果
    if not aggregate_results():
        print("❌ 结果聚合失败")
        sys.exit(1)
    
    # 7. 显示结果
    end_time = time.time()
    print(f"\n⏱️  总耗时: {end_time - start_time:.2f} 秒")
    show_results()
    
    print("\n🎉 压测完成！")
    print("📁 详细结果请查看 results/ 目录")
    print("📊 聚合结果请查看 results/aggregate_results.csv")

if __name__ == "__main__":
    main()