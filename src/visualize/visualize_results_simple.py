#!/usr/bin/env python3
# 文件: visualize_results_simple.py
# 功能: vLLM压测结果简单可视化工具（无外部依赖版本）
# 说明: 根据聚合结果CSV文件生成关键性能指标的文本报告和简单图表

import csv
import os
import glob
import argparse
from datetime import datetime

def load_csv_data(csv_path):
    """
    加载CSV文件数据
    
    参数:
        csv_path (str): CSV文件路径
        
    返回:
        list: 包含字典的列表，每个字典代表一行数据
    """
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # 跳过中文列名行
        next(reader)
        for row in reader:
            # 转换数值字段
            try:
                row['total_input_tokens'] = int(float(row['total_input_tokens']))
                row['total_output_tokens'] = int(float(row['total_output_tokens']))
                row['total_token_throughput'] = float(row['total_token_throughput'])
                row['output_throughput'] = float(row['output_throughput'])
                row['mean_ttft_ms'] = float(row['mean_ttft_ms'])
                row['mean_tpot_ms'] = float(row['mean_tpot_ms'])
                row['mean_itl_ms'] = float(row['mean_itl_ms'])
                row['request_throughput'] = float(row['request_throughput'])
                row['max_concurrency'] = int(row['max_concurrency'])
                row['input_len'] = int(row['input_len']) if row['input_len'] else 0
                row['output_len'] = int(row['output_len']) if row['output_len'] else 0
                
                # 添加配置标识
                row['config'] = f"{row['input_len']}x{row['output_len']}_mc{row['max_concurrency']}"
                
                data.append(row)
            except (ValueError, TypeError) as e:
                print(f"跳过无效数据行: {e}")
                continue
    
    return data

def create_text_report(data, output_dir):
    """
    创建详细的文本性能报告
    """
    report_path = os.path.join(output_dir, 'performance_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("vLLM 性能测试报告 / vLLM Performance Test Report\n")
        f.write("=" * 80 + "\n")
        f.write(f"生成时间 / Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"测试配置数量 / Test Configurations: {len(data)}\n\n")
        
        # 找出最佳性能配置
        if data:
            best_throughput = max(data, key=lambda x: x['total_token_throughput'])
            best_ttft = min(data, key=lambda x: x['mean_ttft_ms'])
            best_tpot = min(data, key=lambda x: x['mean_tpot_ms'])
            
            f.write("最佳性能配置 / Best Performance Configurations:\n")
            f.write("-" * 60 + "\n")
            f.write(f"最高总吞吐量 / Highest Total Throughput:\n")
            f.write(f"  配置: {best_throughput['config']}\n")
            f.write(f"  吞吐量: {best_throughput['total_token_throughput']:.1f} tokens/s\n\n")
            
            f.write(f"最低首令牌时间 / Lowest TTFT:\n")
            f.write(f"  配置: {best_ttft['config']}\n")
            f.write(f"  TTFT: {best_ttft['mean_ttft_ms']:.1f} ms\n\n")
            
            f.write(f"最低每令牌时间 / Lowest TPOT:\n")
            f.write(f"  配置: {best_tpot['config']}\n")
            f.write(f"  TPOT: {best_tpot['mean_tpot_ms']:.1f} ms\n\n")
        
        # 详细性能数据表格
        f.write("详细性能数据 / Detailed Performance Data:\n")
        f.write("=" * 80 + "\n")
        
        # 表头
        f.write(f"{'配置':<15} {'总吞吐量':<12} {'生成吞吐量':<12} {'TTFT':<8} {'TPOT':<8} {'ITL':<8} {'请求吞吐量':<12}\n")
        f.write(f"{'Config':<15} {'Total Tput':<12} {'Gen Tput':<12} {'(ms)':<8} {'(ms)':<8} {'(ms)':<8} {'Req Tput':<12}\n")
        f.write("-" * 80 + "\n")
        
        for row in data:
            f.write(f"{row['config']:<15} "
                   f"{row['total_token_throughput']:<12.1f} "
                   f"{row['output_throughput']:<12.1f} "
                   f"{row['mean_ttft_ms']:<8.1f} "
                   f"{row['mean_tpot_ms']:<8.1f} "
                   f"{row['mean_itl_ms']:<8.1f} "
                   f"{row['request_throughput']:<12.3f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        
        # 令牌统计
        f.write("令牌统计 / Token Statistics:\n")
        f.write("-" * 40 + "\n")
        for row in data:
            f.write(f"配置 {row['config']}:\n")
            f.write(f"  输入令牌 / Input Tokens: {row['total_input_tokens']:,}\n")
            f.write(f"  输出令牌 / Output Tokens: {row['total_output_tokens']:,}\n")
            f.write(f"  总令牌 / Total Tokens: {row['total_input_tokens'] + row['total_output_tokens']:,}\n")
            f.write("\n")
    
    print(f"详细性能报告已保存到: {report_path}")

def create_summary_table(data, output_dir):
    """
    创建性能摘要表格（CSV格式）
    """
    summary_path = os.path.join(output_dir, 'performance_summary.csv')
    
    with open(summary_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        
        # 写入表头（中英文）
        writer.writerow([
            '配置/Config',
            '输入令牌/Input Tokens', 
            '输出令牌/Output Tokens',
            '总吞吐量/Total Throughput (tok/s)',
            '生成吞吐量/Gen Throughput (tok/s)',
            'TTFT (ms)',
            'TPOT (ms)', 
            'ITL (ms)',
            '请求吞吐量/Req Throughput (req/s)'
        ])
        
        # 写入数据
        for row in data:
            writer.writerow([
                row['config'],
                row['total_input_tokens'],
                row['total_output_tokens'],
                f"{row['total_token_throughput']:.1f}",
                f"{row['output_throughput']:.1f}",
                f"{row['mean_ttft_ms']:.1f}",
                f"{row['mean_tpot_ms']:.1f}",
                f"{row['mean_itl_ms']:.1f}",
                f"{row['request_throughput']:.3f}"
            ])
    
    print(f"性能摘要表格已保存到: {summary_path}")

def create_ascii_charts(data, output_dir):
    """
    创建ASCII字符图表
    """
    chart_path = os.path.join(output_dir, 'ascii_charts.txt')
    
    with open(chart_path, 'w', encoding='utf-8') as f:
        f.write("ASCII 性能图表 / ASCII Performance Charts\n")
        f.write("=" * 60 + "\n\n")
        
        # 吞吐量条形图
        f.write("总令牌吞吐量对比 / Total Token Throughput Comparison:\n")
        f.write("-" * 50 + "\n")
        
        if data:
            max_throughput = max(row['total_token_throughput'] for row in data)
            
            for row in data:
                throughput = row['total_token_throughput']
                bar_length = int((throughput / max_throughput) * 40)
                bar = "█" * bar_length
                f.write(f"{row['config']:<15} {bar:<40} {throughput:.1f} tok/s\n")
        
        f.write("\n")
        
        # TTFT条形图
        f.write("首令牌时间对比 / TTFT Comparison (越短越好/Shorter is Better):\n")
        f.write("-" * 50 + "\n")
        
        if data:
            max_ttft = max(row['mean_ttft_ms'] for row in data)
            
            for row in data:
                ttft = row['mean_ttft_ms']
                bar_length = int((ttft / max_ttft) * 40)
                bar = "█" * bar_length
                f.write(f"{row['config']:<15} {bar:<40} {ttft:.1f} ms\n")
        
        f.write("\n")
        
        # TPOT条形图
        f.write("每令牌时间对比 / TPOT Comparison (越短越好/Shorter is Better):\n")
        f.write("-" * 50 + "\n")
        
        if data:
            max_tpot = max(row['mean_tpot_ms'] for row in data)
            
            for row in data:
                tpot = row['mean_tpot_ms']
                bar_length = int((tpot / max_tpot) * 40)
                bar = "█" * bar_length
                f.write(f"{row['config']:<15} {bar:<40} {tpot:.1f} ms\n")
    
    print(f"ASCII图表已保存到: {chart_path}")

def create_performance_ranking(data, output_dir):
    """
    创建性能排名
    """
    ranking_path = os.path.join(output_dir, 'performance_ranking.txt')
    
    with open(ranking_path, 'w', encoding='utf-8') as f:
        f.write("性能排名 / Performance Ranking\n")
        f.write("=" * 50 + "\n\n")
        
        # 按总吞吐量排名
        f.write("按总吞吐量排名 / Ranking by Total Throughput:\n")
        f.write("-" * 40 + "\n")
        throughput_sorted = sorted(data, key=lambda x: x['total_token_throughput'], reverse=True)
        for i, row in enumerate(throughput_sorted, 1):
            f.write(f"{i}. {row['config']:<15} {row['total_token_throughput']:.1f} tok/s\n")
        
        f.write("\n")
        
        # 按TTFT排名（越小越好）
        f.write("按TTFT排名 / Ranking by TTFT (越小越好/Lower is Better):\n")
        f.write("-" * 40 + "\n")
        ttft_sorted = sorted(data, key=lambda x: x['mean_ttft_ms'])
        for i, row in enumerate(ttft_sorted, 1):
            f.write(f"{i}. {row['config']:<15} {row['mean_ttft_ms']:.1f} ms\n")
        
        f.write("\n")
        
        # 按TPOT排名（越小越好）
        f.write("按TPOT排名 / Ranking by TPOT (越小越好/Lower is Better):\n")
        f.write("-" * 40 + "\n")
        tpot_sorted = sorted(data, key=lambda x: x['mean_tpot_ms'])
        for i, row in enumerate(tpot_sorted, 1):
            f.write(f"{i}. {row['config']:<15} {row['mean_tpot_ms']:.1f} ms\n")
        
        f.write("\n")
        
        # 综合评分（简单加权）
        f.write("综合评分排名 / Overall Score Ranking:\n")
        f.write("(评分方法: 吞吐量权重0.5 + TTFT权重0.25 + TPOT权重0.25)\n")
        f.write("-" * 40 + "\n")
        
        # 标准化并计算综合评分
        if data:
            max_throughput = max(row['total_token_throughput'] for row in data)
            min_ttft = min(row['mean_ttft_ms'] for row in data)
            max_ttft = max(row['mean_ttft_ms'] for row in data)
            min_tpot = min(row['mean_tpot_ms'] for row in data)
            max_tpot = max(row['mean_tpot_ms'] for row in data)
            
            scored_data = []
            for row in data:
                # 标准化分数 (0-1)
                throughput_score = row['total_token_throughput'] / max_throughput
                ttft_score = 1 - (row['mean_ttft_ms'] - min_ttft) / (max_ttft - min_ttft) if max_ttft > min_ttft else 1
                tpot_score = 1 - (row['mean_tpot_ms'] - min_tpot) / (max_tpot - min_tpot) if max_tpot > min_tpot else 1
                
                # 加权综合评分
                overall_score = throughput_score * 0.5 + ttft_score * 0.25 + tpot_score * 0.25
                
                scored_data.append((row, overall_score))
            
            # 按综合评分排序
            scored_data.sort(key=lambda x: x[1], reverse=True)
            
            for i, (row, score) in enumerate(scored_data, 1):
                f.write(f"{i}. {row['config']:<15} 评分: {score:.3f}\n")
    
    print(f"性能排名已保存到: {ranking_path}")

def find_latest_csv():
    """
    查找最新的聚合结果CSV文件
    """
    csv_files = glob.glob("results/*/aggregate_results_*.csv")
    if not csv_files:
        csv_files = glob.glob("results/aggregate_results_*.csv")
    
    if not csv_files:
        return None
    
    # 按修改时间排序，返回最新的
    csv_files.sort(key=os.path.getmtime, reverse=True)
    return csv_files[0]

def main():
    parser = argparse.ArgumentParser(
        description="vLLM压测结果简单可视化工具（无外部依赖版本）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python visualize_results_simple.py                                    # 自动查找最新的CSV文件
  python visualize_results_simple.py --csv results/aggregate_results_20250728.csv
  python visualize_results_simple.py --csv results/DeepSeek-R1_20250728_152452/aggregate_results_20250728.csv --output reports
        """
    )
    
    parser.add_argument('--csv', help='聚合结果CSV文件路径（不指定则自动查找最新的）')
    parser.add_argument('--output', default='reports', help='输出目录（默认: reports）')
    
    args = parser.parse_args()
    
    # 确定CSV文件路径
    if args.csv:
        csv_path = args.csv
    else:
        csv_path = find_latest_csv()
        if not csv_path:
            print("错误: 未找到聚合结果CSV文件")
            print("请先运行 'python main.py aggregate' 生成聚合结果")
            return
        print(f"使用最新的CSV文件: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"错误: CSV文件不存在: {csv_path}")
        return
    
    # 创建输出目录
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"正在加载数据: {csv_path}")
    data = load_csv_data(csv_path)
    
    if not data:
        print("错误: 未能加载有效数据")
        return
    
    print(f"找到 {len(data)} 个测试配置")
    print("正在生成报告...")
    
    # 生成各种报告
    create_text_report(data, output_dir)
    print("✓ 详细性能报告已生成")
    
    create_summary_table(data, output_dir)
    print("✓ 性能摘要表格已生成")
    
    create_ascii_charts(data, output_dir)
    print("✓ ASCII图表已生成")
    
    create_performance_ranking(data, output_dir)
    print("✓ 性能排名已生成")
    
    print(f"\n所有报告文件已保存到: {output_dir}/")
    print("生成的文件:")
    print("  - performance_report.txt     (详细性能报告)")
    print("  - performance_summary.csv    (性能摘要表格)")
    print("  - ascii_charts.txt           (ASCII图表)")
    print("  - performance_ranking.txt    (性能排名)")
    
    # 显示关键指标摘要
    print(f"\n关键指标摘要 / Key Metrics Summary:")
    print("=" * 60)
    print(f"{'配置':<15} {'总吞吐量':<12} {'TTFT':<8} {'TPOT':<8} {'ITL':<8}")
    print(f"{'Config':<15} {'(tok/s)':<12} {'(ms)':<8} {'(ms)':<8} {'(ms)':<8}")
    print("-" * 60)
    for row in data:
        print(f"{row['config']:<15} "
              f"{row['total_token_throughput']:<12.1f} "
              f"{row['mean_ttft_ms']:<8.1f} "
              f"{row['mean_tpot_ms']:<8.1f} "
              f"{row['mean_itl_ms']:<8.1f}")

if __name__ == "__main__":
    main()