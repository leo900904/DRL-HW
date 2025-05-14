#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
结果比较脚本
用于比较不同配置的Dueling DQN训练结果
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import argparse
import glob
import re
from dueling_dqn_lightning import DuelingDQNLightning, test_model
from Gridworld import Gridworld

# 设置中文字体支持
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
except:
    print("警告: 无法设置中文字体，图表中的中文可能显示为方框")

def load_results(results_file):
    """
    从结果文件加载训练数据
    """
    data = {}
    try:
        with open(results_file, 'r') as f:
            for line in f:
                key, value = line.strip().split(':')
                data[key.strip()] = float(value.strip().replace('%', ''))
        return data
    except Exception as e:
        print(f"加载结果文件失败: {e}")
        return None

def compare_models_from_results(results_files, output_dir='comparison_plots'):
    """
    比较不同结果文件中的模型性能
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    all_results = []
    labels = []
    
    # 加载所有结果
    for result_file in results_files:
        data = load_results(result_file)
        if data:
            all_results.append(data)
            # 从文件名提取标签
            label = os.path.basename(result_file).replace('results_', '').replace('.txt', '')
            labels.append(label)
    
    if not all_results:
        print("没有找到有效的结果文件")
        return
    
    # 提取所有指标
    metrics = list(all_results[0].keys())
    
    # 为每个指标创建比较图
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        values = [result.get(metric, 0) for result in all_results]
        
        # 创建条形图
        bars = plt.bar(range(len(labels)), values, color='skyblue', edgecolor='navy')
        
        # 在条形上方添加数值标签
        for i, v in enumerate(values):
            if metric.endswith('率'):
                plt.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=12)
            else:
                plt.text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=12)
        
        plt.xlabel('训练配置', fontsize=14)
        plt.ylabel(metric, fontsize=14)
        plt.title(f'不同配置的{metric}比较', fontsize=16, fontweight='bold')
        plt.xticks(range(len(labels)), labels, rotation=45)
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 保存图表
        plt.savefig(os.path.join(output_dir, f'comparison_{metric}.png'), dpi=300)
        plt.close()
    
    # 创建所有指标的组合比较图
    plt.figure(figsize=(15, 10))
    x = np.arange(len(labels))
    width = 0.2
    offsets = np.linspace(-0.3, 0.3, len(metrics))
    
    for i, metric in enumerate(metrics):
        values = [result.get(metric, 0) for result in all_results]
        # 将所有指标标准化到0-1范围，以便在同一图表上显示
        max_val = max(values) if max(values) > 0 else 1
        normalized_values = [v / max_val for v in values]
        plt.bar(x + offsets[i], normalized_values, width, label=metric)
    
    plt.xlabel('训练配置', fontsize=14)
    plt.ylabel('标准化值', fontsize=14)
    plt.title('所有指标的比较 (标准化)', fontsize=16, fontweight='bold')
    plt.xticks(x, labels, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'comparison_all_metrics.png'), dpi=300)
    plt.close()
    
    print(f"比较图表已保存到 {output_dir} 目录")

def test_all_models(model_dir='models', output_dir='model_tests', mode='random', episodes=100):
    """
    测试指定目录中的所有模型文件
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 查找所有模型文件
    model_files = glob.glob(os.path.join(model_dir, '*.pth'))
    if not model_files:
        print(f"在 {model_dir} 目录中没有找到模型文件")
        return
    
    # 对文件名进行排序，确保epoch顺序一致
    model_files.sort(key=lambda x: int(re.search(r'epoch_(\d+)', x).group(1)) if re.search(r'epoch_(\d+)', x) else float('inf'))
    
    # 保存每个模型的测试结果
    success_rates = []
    model_labels = []
    
    for model_file in model_files:
        # 从文件名提取标签/轮次
        if 'epoch_' in model_file:
            epoch = re.search(r'epoch_(\d+)', model_file).group(1)
            label = f'Epoch {epoch}'
        else:
            label = os.path.basename(model_file).replace('.pth', '')
        
        print(f"测试模型: {model_file}")
        model = DuelingDQNLightning()
        try:
            model.load_state_dict(torch.load(model_file))
            model.eval()
            
            # 测试模型
            success_rate = test_model(model, Gridworld, mode=mode, test_episodes=episodes)
            success_rates.append(success_rate)
            model_labels.append(label)
            
            print(f"模型 {label} 成功率: {success_rate:.1f}%")
            
            # 保存单独的测试结果
            with open(os.path.join(output_dir, f'test_{label}.txt'), 'w') as f:
                f.write(f"模型: {model_file}\n")
                f.write(f"测试模式: {mode}\n")
                f.write(f"测试轮数: {episodes}\n")
                f.write(f"成功率: {success_rate:.1f}%\n")
        
        except Exception as e:
            print(f"测试模型 {model_file} 失败: {e}")
    
    # 创建成功率随时间变化的图表
    plt.figure(figsize=(12, 8))
    plt.plot(model_labels, success_rates, 'o-', linewidth=2, markersize=8)
    plt.xlabel('训练进度', fontsize=14)
    plt.ylabel('成功率 (%)', fontsize=14)
    plt.title(f'模型成功率随训练进度变化 (模式: {mode})', fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 105)  # 确保y轴从0到100以上
    plt.xticks(rotation=45)
    
    # 标记最高成功率点
    max_idx = np.argmax(success_rates)
    max_val = success_rates[max_idx]
    plt.scatter([model_labels[max_idx]], [max_val], color='red', s=100, zorder=5)
    plt.annotate(f'最高成功率: {max_val:.1f}%',
                xy=(model_labels[max_idx], max_val),
                xytext=(0, 20),
                textcoords='offset points',
                ha='center',
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'success_rate_progression_{mode}.png'), dpi=300)
    plt.close()
    
    # 创建综合结果文件
    with open(os.path.join(output_dir, 'all_test_results.txt'), 'w') as f:
        f.write(f"测试模式: {mode}\n")
        f.write(f"测试轮数: {episodes}\n\n")
        f.write("测试结果汇总:\n")
        for i, (label, rate) in enumerate(zip(model_labels, success_rates)):
            f.write(f"{i+1}. {label}: {rate:.1f}%\n")
        f.write(f"\n最高成功率: {max(success_rates):.1f}% (模型: {model_labels[np.argmax(success_rates)]})")
    
    print(f"测试结果已保存到 {output_dir} 目录")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dueling DQN训练结果比较与模型测试工具')
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # 比较结果的子命令
    compare_parser = subparsers.add_parser('compare', help='比较不同训练结果')
    compare_parser.add_argument('--results', nargs='+', default=None, 
                               help='要比较的结果文件列表，如果不提供，将使用所有results_*.txt文件')
    compare_parser.add_argument('--output_dir', default='comparison_plots',
                               help='比较图表的输出目录')
    
    # 测试模型的子命令
    test_parser = subparsers.add_parser('test', help='测试保存的模型')
    test_parser.add_argument('--model_dir', default='models',
                            help='包含模型文件的目录')
    test_parser.add_argument('--output_dir', default='model_tests',
                            help='测试结果的输出目录')
    test_parser.add_argument('--mode', choices=['static', 'random', 'player'], default='random',
                            help='测试环境模式')
    test_parser.add_argument('--episodes', type=int, default=100,
                            help='每个模型测试的轮数')
    
    args = parser.parse_args()
    
    if args.command == 'compare':
        results_files = args.results
        if results_files is None:
            # 如果未提供结果文件，使用所有results_*.txt文件
            results_files = glob.glob('results_*.txt')
        
        compare_models_from_results(results_files, args.output_dir)
    
    elif args.command == 'test':
        test_all_models(args.model_dir, args.output_dir, args.mode, args.episodes)
    
    else:
        parser.print_help() 