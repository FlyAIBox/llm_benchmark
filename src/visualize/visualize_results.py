#!/usr/bin/env python3
# 文件: visualize_results.py
# 功能: vLLM压测结果可视化工具
# 说明: 根据聚合结果CSV文件生成关键性能指标的可视化图表

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
import glob
from datetime import datetime

# 设置中文字体支持
import matplotlib.font_manager as fm

# 查找可用的中文字体
def setup_chinese_font():
    """设置中文字体支持"""
    import os
    import shutil
    
    # 清理matplotlib字体缓存
    try:
        cache_dir = os.path.join(os.path.expanduser('~'), '.matplotlib')
        if os.path.exists(cache_dir):
            print("清理matplotlib字体缓存...")
            shutil.rmtree(cache_dir, ignore_errors=True)
    except Exception as e:
        print(f"清理缓存失败（忽略）: {e}")
    
    # 重新加载字体管理器
    fm.fontManager.__init__()

    # 直接指定字体文件路径，优先使用字符集更完整的中文字体
    font_files = [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',  # Noto CJK - 优先
        '/usr/share/fonts/truetype/arphic/uming.ttc',      # AR PL UMing CN
        '/usr/share/fonts/truetype/arphic/ukai.ttc',       # AR PL UKai CN
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',  # WenQuanYi Micro Hei
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',    # WenQuanYi Zen Hei
    ]

    # 尝试添加字体文件
    selected_font = None
    available_chinese_fonts = []
    
    for font_file in font_files:
        if os.path.exists(font_file):
            try:
                # 添加字体到matplotlib
                fm.fontManager.addfont(font_file)

                # 获取字体名称
                for font in fm.fontManager.ttflist:
                    if font.fname == font_file:
                        font_name = font.name
                        if any(keyword in font_name for keyword in ['CJK', 'CN', 'UMing', 'UKai', 'WenQuanYi']):
                            available_chinese_fonts.append(font_name)
                            if not selected_font:
                                selected_font = font_name
                                print(f"✓ 成功添加中文字体: {selected_font} ({font_file})")
            except Exception as e:
                print(f"✗ 添加字体失败 {font_file}: {e}")

    # 备用字体名称列表（系统预装的字体）
    fallback_fonts = [
        'Noto Sans CJK SC',
        'Noto Serif CJK SC',
        'AR PL UMing CN',
        'AR PL UKai CN', 
        'WenQuanYi Micro Hei',
        'WenQuanYi Zen Hei',
        'SimHei',
        'Microsoft YaHei',
        'PingFang SC',
        'Hiragino Sans GB'
    ]

    # 如果直接添加字体失败，尝试使用系统字体名称
    if not selected_font:
        print("尝试查找系统预装的中文字体...")
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        for font in fallback_fonts:
            if font in available_fonts:
                selected_font = font
                print(f"✓ 使用系统中文字体: {selected_font}")
                break

    # 强制设置字体参数
    if selected_font:
        # 设置多个备选字体确保覆盖
        font_list = [selected_font] + available_chinese_fonts + ['DejaVu Sans', 'Arial', 'Liberation Sans']
        # 去重
        font_list = list(dict.fromkeys(font_list))
        
        plt.rcParams['font.sans-serif'] = font_list
        plt.rcParams['font.family'] = 'sans-serif'
        
        # 强制设置所有相关的字体参数
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.titlesize'] = 'large'
        
        print(f"✓ 字体设置完成，字体列表: {font_list[:3]}...")
        
        # 验证字体设置
        current_font = plt.rcParams['font.sans-serif'][0]
        print(f"✓ 当前使用字体: {current_font}")
        
    else:
        print("⚠️  警告: 未找到中文字体，图表中的中文可能显示为方框")
        print("建议安装字体: sudo apt-get install fonts-noto-cjk")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
        plt.rcParams['axes.unicode_minus'] = False

    return selected_font

# 初始化字体设置
font_status = setup_chinese_font()

# 设置图表样式
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

def load_aggregate_data(csv_path):
    """
    加载聚合结果CSV文件
    
    参数:
        csv_path (str): CSV文件路径
        
    返回:
        DataFrame: 处理后的数据
    """
    # 读取CSV文件，跳过中文列名行
    df = pd.read_csv(csv_path, skiprows=[1])
    
    # 计算关键指标
    df['prompt_tokens'] = df['total_input_tokens']
    df['completion_tokens'] = df['total_output_tokens'] 
    df['TOTAL_THROUGHPUT'] = df['total_token_throughput']
    df['generate_throughput'] = df['output_throughput']
    df['TTFT'] = df['mean_ttft_ms']
    df['TPOT'] = df['mean_tpot_ms']
    df['ITL'] = df['mean_itl_ms']
    
    # 添加配置标识
    df['config'] = df['input_len'].astype(str) + 'x' + df['output_len'].astype(str) + '_mc' + df['max_concurrency'].astype(str)
    
    return df

def create_throughput_comparison(df, output_dir):
    """
    创建吞吐量对比图表
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('吞吐量性能对比 / Throughput Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1. 总吞吐量对比
    bars1 = ax1.bar(range(len(df)), df['TOTAL_THROUGHPUT'], 
                    color='skyblue', alpha=0.8, edgecolor='navy')
    ax1.set_title('总令牌吞吐量 / Total Token Throughput (tokens/s)', fontweight='bold')
    ax1.set_ylabel('吞吐量 (tokens/s) / Throughput (tokens/s)')
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels(df['config'], rotation=45, ha='right')
    
    # 添加数值标签
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. 生成吞吐量对比
    bars2 = ax2.bar(range(len(df)), df['generate_throughput'], 
                    color='lightgreen', alpha=0.8, edgecolor='darkgreen')
    ax2.set_title('生成令牌吞吐量 / Generation Throughput (tokens/s)', fontweight='bold')
    ax2.set_ylabel('吞吐量 (tokens/s) / Throughput (tokens/s)')
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels(df['config'], rotation=45, ha='right')
    
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. 输入输出令牌数对比
    x = np.arange(len(df))
    width = 0.35
    
    bars3 = ax3.bar(x - width/2, df['prompt_tokens'], width, 
                    label='输入令牌 / Input Tokens', color='orange', alpha=0.8)
    bars4 = ax3.bar(x + width/2, df['completion_tokens'], width,
                    label='输出令牌 / Output Tokens', color='purple', alpha=0.8)
    
    ax3.set_title('令牌数量对比 / Token Count Comparison', fontweight='bold')
    ax3.set_ylabel('令牌数量 / Token Count')
    ax3.set_xticks(x)
    ax3.set_xticklabels(df['config'], rotation=45, ha='right')
    ax3.legend()
    
    # 4. 请求吞吐量
    bars5 = ax4.bar(range(len(df)), df['request_throughput'], 
                    color='coral', alpha=0.8, edgecolor='darkred')
    ax4.set_title('请求吞吐量 / Request Throughput (req/s)', fontweight='bold')
    ax4.set_ylabel('吞吐量 (req/s) / Throughput (req/s)')
    ax4.set_xticks(range(len(df)))
    ax4.set_xticklabels(df['config'], rotation=45, ha='right')
    
    for i, bar in enumerate(bars5):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'throughput_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_latency_comparison(df, output_dir):
    """
    创建延迟性能对比图表
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('延迟性能对比 / Latency Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1. TTFT对比
    bars1 = ax1.bar(range(len(df)), df['TTFT'], 
                    color='lightcoral', alpha=0.8, edgecolor='darkred')
    ax1.set_title('首令牌时间 / Time to First Token (ms)', fontweight='bold')
    ax1.set_ylabel('延迟 (ms) / Latency (ms)')
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels(df['config'], rotation=45, ha='right')
    
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. TPOT对比
    bars2 = ax2.bar(range(len(df)), df['TPOT'], 
                    color='gold', alpha=0.8, edgecolor='orange')
    ax2.set_title('每令牌时间 / Time per Output Token (ms)', fontweight='bold')
    ax2.set_ylabel('延迟 (ms) / Latency (ms)')
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels(df['config'], rotation=45, ha='right')
    
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. ITL对比
    bars3 = ax3.bar(range(len(df)), df['ITL'], 
                    color='lightblue', alpha=0.8, edgecolor='blue')
    ax3.set_title('令牌间延迟 / Inter-token Latency (ms)', fontweight='bold')
    ax3.set_ylabel('延迟 (ms) / Latency (ms)')
    ax3.set_xticks(range(len(df)))
    ax3.set_xticklabels(df['config'], rotation=45, ha='right')
    
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. 端到端延迟对比
    bars4 = ax4.bar(range(len(df)), df['mean_e2el_ms'], 
                    color='plum', alpha=0.8, edgecolor='purple')
    ax4.set_title('端到端延迟 / End-to-End Latency (ms)', fontweight='bold')
    ax4.set_ylabel('延迟 (ms) / Latency (ms)')
    ax4.set_xticks(range(len(df)))
    ax4.set_xticklabels(df['config'], rotation=45, ha='right')
    
    for i, bar in enumerate(bars4):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latency_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_heatmap(df, output_dir):
    """
    创建性能热力图
    """
    # 准备热力图数据
    metrics = ['TOTAL_THROUGHPUT', 'generate_throughput', 'TTFT', 'TPOT', 'ITL']
    metric_names = ['总吞吐量\nTotal Throughput', '生成吞吐量\nGeneration Throughput', 
                   '首令牌时间\nTTFT', '每令牌时间\nTPOT', '令牌间延迟\nITL']
    
    # 标准化数据用于热力图显示
    heatmap_data = df[metrics].copy()
    
    # 对延迟指标取倒数，使得越小的值在热力图中显示为越好（颜色越深）
    for col in ['TTFT', 'TPOT', 'ITL']:
        heatmap_data[col] = 1 / heatmap_data[col] * 1000  # 转换为倒数并放大
    
    # 标准化到0-1范围
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    heatmap_normalized = pd.DataFrame(
        scaler.fit_transform(heatmap_data),
        columns=metric_names,
        index=df['config']
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_normalized.T, annot=True, cmap='RdYlGn', 
                cbar_kws={'label': '性能得分 / Performance Score (0-1)'})
    plt.title('性能热力图 / Performance Heatmap\n(绿色=更好 / Green=Better)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('配置 / Configuration')
    plt.ylabel('性能指标 / Performance Metrics')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_comprehensive_dashboard(df, output_dir):
    """
    创建综合性能仪表板
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 主标题
    fig.suptitle('vLLM 性能测试综合仪表板 / vLLM Performance Testing Dashboard', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    # 1. 吞吐量对比 (左上)
    ax1 = fig.add_subplot(gs[0, :2])
    bars = ax1.bar(range(len(df)), df['TOTAL_THROUGHPUT'], 
                   color='skyblue', alpha=0.8, edgecolor='navy')
    ax1.set_title('总令牌吞吐量 / Total Token Throughput', fontweight='bold')
    ax1.set_ylabel('tokens/s')
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels(df['config'], rotation=45, ha='right')
    
    # 2. 延迟对比 (右上)
    ax2 = fig.add_subplot(gs[0, 2:])
    x = np.arange(len(df))
    width = 0.25
    ax2.bar(x - width, df['TTFT'], width, label='TTFT', alpha=0.8)
    ax2.bar(x, df['TPOT'], width, label='TPOT', alpha=0.8)
    ax2.bar(x + width, df['ITL'], width, label='ITL', alpha=0.8)
    ax2.set_title('延迟指标对比 / Latency Metrics', fontweight='bold')
    ax2.set_ylabel('ms')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['config'], rotation=45, ha='right')
    ax2.legend()
    
    # 3. 令牌数量 (中左)
    ax3 = fig.add_subplot(gs[1, :2])
    x = np.arange(len(df))
    width = 0.35
    ax3.bar(x - width/2, df['prompt_tokens'], width, 
            label='输入令牌 / Input', alpha=0.8)
    ax3.bar(x + width/2, df['completion_tokens'], width,
            label='输出令牌 / Output', alpha=0.8)
    ax3.set_title('令牌数量 / Token Counts', fontweight='bold')
    ax3.set_ylabel('Token Count')
    ax3.set_xticks(x)
    ax3.set_xticklabels(df['config'], rotation=45, ha='right')
    ax3.legend()
    
    # 4. 请求吞吐量 (中右)
    ax4 = fig.add_subplot(gs[1, 2:])
    bars = ax4.bar(range(len(df)), df['request_throughput'], 
                   color='coral', alpha=0.8, edgecolor='darkred')
    ax4.set_title('请求吞吐量 / Request Throughput', fontweight='bold')
    ax4.set_ylabel('req/s')
    ax4.set_xticks(range(len(df)))
    ax4.set_xticklabels(df['config'], rotation=45, ha='right')
    
    # 5. 性能摘要表格 (底部)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('tight')
    ax5.axis('off')
    
    # 创建摘要表格数据
    summary_data = []
    for _, row in df.iterrows():
        summary_data.append([
            row['config'],
            f"{row['TOTAL_THROUGHPUT']:.1f}",
            f"{row['generate_throughput']:.1f}",
            f"{row['TTFT']:.1f}",
            f"{row['TPOT']:.1f}",
            f"{row['ITL']:.1f}",
            f"{row['request_throughput']:.3f}"
        ])
    
    headers = ['配置\nConfig', '总吞吐量\nTotal Tput\n(tok/s)', '生成吞吐量\nGen Tput\n(tok/s)',
               'TTFT\n(ms)', 'TPOT\n(ms)', 'ITL\n(ms)', '请求吞吐量\nReq Tput\n(req/s)']
    
    table = ax5.table(cellText=summary_data, colLabels=headers,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # 设置表格样式
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.savefig(os.path.join(output_dir, 'comprehensive_dashboard.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def generate_performance_report(df, output_dir):
    """
    生成性能报告文本
    """
    report_path = os.path.join(output_dir, 'performance_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("vLLM 性能测试报告 / vLLM Performance Test Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"生成时间 / Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"测试配置数量 / Test Configurations: {len(df)}\n\n")
        
        # 最佳性能配置
        best_throughput_idx = df['TOTAL_THROUGHPUT'].idxmax()
        best_ttft_idx = df['TTFT'].idxmin()
        best_tpot_idx = df['TPOT'].idxmin()
        
        f.write("最佳性能配置 / Best Performance Configurations:\n")
        f.write("-" * 50 + "\n")
        f.write(f"最高吞吐量 / Highest Throughput: {df.loc[best_throughput_idx, 'config']} "
                f"({df.loc[best_throughput_idx, 'TOTAL_THROUGHPUT']:.1f} tokens/s)\n")
        f.write(f"最低TTFT / Lowest TTFT: {df.loc[best_ttft_idx, 'config']} "
                f"({df.loc[best_ttft_idx, 'TTFT']:.1f} ms)\n")
        f.write(f"最低TPOT / Lowest TPOT: {df.loc[best_tpot_idx, 'config']} "
                f"({df.loc[best_tpot_idx, 'TPOT']:.1f} ms)\n\n")
        
        # 详细性能数据
        f.write("详细性能数据 / Detailed Performance Data:\n")
        f.write("-" * 50 + "\n")
        for _, row in df.iterrows():
            f.write(f"配置 / Config: {row['config']}\n")
            f.write(f"  总吞吐量 / Total Throughput: {row['TOTAL_THROUGHPUT']:.1f} tokens/s\n")
            f.write(f"  生成吞吐量 / Generation Throughput: {row['generate_throughput']:.1f} tokens/s\n")
            f.write(f"  请求吞吐量 / Request Throughput: {row['request_throughput']:.3f} req/s\n")
            f.write(f"  TTFT: {row['TTFT']:.1f} ms\n")
            f.write(f"  TPOT: {row['TPOT']:.1f} ms\n")
            f.write(f"  ITL: {row['ITL']:.1f} ms\n")
            f.write(f"  输入令牌 / Input Tokens: {row['prompt_tokens']}\n")
            f.write(f"  输出令牌 / Output Tokens: {row['completion_tokens']}\n")
            f.write("\n")
    
    print(f"性能报告已保存到: {report_path}")

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
        description="vLLM压测结果可视化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python visualize_results.py                                    # 自动查找最新的CSV文件
  python visualize_results.py --csv results/aggregate_results_20250728.csv
  python visualize_results.py --csv results/DeepSeek-R1_20250728_152452/aggregate_results_20250728.csv --output charts
        """
    )
    
    parser.add_argument('--csv', help='聚合结果CSV文件路径（不指定则自动查找最新的）')
    parser.add_argument('--output', default='charts', help='输出目录（默认: charts）')
    
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
    df = load_aggregate_data(csv_path)
    
    print(f"找到 {len(df)} 个测试配置")
    print("正在生成可视化图表...")
    
    # 生成各种图表
    create_throughput_comparison(df, output_dir)
    print("✓ 吞吐量对比图表已生成")
    
    create_latency_comparison(df, output_dir)
    print("✓ 延迟对比图表已生成")
    
    create_performance_heatmap(df, output_dir)
    print("✓ 性能热力图已生成")
    
    create_comprehensive_dashboard(df, output_dir)
    print("✓ 综合仪表板已生成")
    
    generate_performance_report(df, output_dir)
    print("✓ 性能报告已生成")
    
    print(f"\n所有可视化文件已保存到: {output_dir}/")
    print("生成的文件:")
    print("  - throughput_comparison.png    (吞吐量对比)")
    print("  - latency_comparison.png       (延迟对比)")
    print("  - performance_heatmap.png      (性能热力图)")
    print("  - comprehensive_dashboard.png  (综合仪表板)")
    print("  - performance_report.txt       (性能报告)")

if __name__ == "__main__":
    main()