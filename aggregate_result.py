# 文件: aggregate_result.py
# 功能: 聚合vLLM推理压测结果，将多个JSON结果文件合并为一个CSV文件
# 依赖: pip install pandas

import glob
import json
import pandas as pd
import os
import re

# 结果文件存储目录
RESULT_DIR = "results"
# 输出的聚合CSV文件路径
OUT_CSV = os.path.join(RESULT_DIR, "aggregate_results.csv")

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

def main():
    """
    主函数：聚合所有压测结果
    
    处理流程:
    1. 扫描results目录下的所有JSON文件
    2. 读取每个JSON文件的内容
    3. 从文件名解析输入输出长度信息
    4. 将所有数据合并到pandas DataFrame
    5. 导出为CSV文件
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

    # 3) 使用pandas将所有记录转换为DataFrame并保存为CSV
    df = pd.json_normalize(records)  # 将嵌套的JSON数据扁平化
    df.to_csv(OUT_CSV, index=False)  # 保存为CSV文件，不包含行索引
    print(f"已聚合 {len(records)} 个测试结果 → {OUT_CSV}")

if __name__ == "__main__":
    main()
