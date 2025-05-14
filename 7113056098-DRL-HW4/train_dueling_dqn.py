#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dueling DQN 训练启动脚本
使用 PyTorch Lightning 重构的 Dueling DQN，包含多种强化学习训练技巧
但不使用Lightning的Trainer，而是使用标准的PyTorch训练循环
增加了详细的训练结果记录和实时绘图功能
"""

import os
# 设置OpenMP环境变量，避免多线程库冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import matplotlib.pyplot as plt
from dueling_dqn_lightning import train_dueling_dqn, test_model, DuelingDQNLightning
from Gridworld import Gridworld
import random
import argparse
import pandas as pd
from datetime import datetime

# 配置matplotlib字体 - 使用英文避免中文显示问题
import matplotlib
# 使用简单的字体和配置，避免警告
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
# 解决负号显示问题
matplotlib.rcParams['axes.unicode_minus'] = False

# 创建日志记录函数
def save_training_log(epochs, rewards, lengths, success_rates, epsilons, losses, filename="training_log.csv"):
    """保存训练日志到CSV文件"""
    data = {
        'Episode': epochs,
        'Reward': rewards,
        'Length': lengths,
        'Success_Rate': success_rates,
        'Epsilon': epsilons,
        'Loss': losses
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"训练日志已保存到 {filename}")

# 实时绘制训练曲线
def plot_live_training(epochs, rewards, lengths, success_rates, epsilons, losses, save_path="plots/live_training"):
    """实时绘制训练曲线并保存"""
    plt.style.use('ggplot')
    os.makedirs(save_path, exist_ok=True)
    
    # 绘制多个指标
    metrics = {
        'reward': (rewards, 'Reward Value', 'Reward'),
        'length': (lengths, 'Episode Length', 'Length'),
        'success_rate': (success_rates, 'Success Rate (%)', 'Success Rate (%)'),
        'epsilon': (epsilons, 'Epsilon Value', 'Epsilon'),
        'loss': (losses, 'Loss Value', 'Loss')
    }
    
    # 单独绘制每个指标
    for name, (data, ylabel, title) in metrics.items():
        if not data:  # 如果数据为空，跳过
            continue
            
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, data, 'b-', linewidth=2)
        
        # 对于损失和奖励，计算移动平均
        if name in ['loss', 'reward'] and len(data) > 5:
            window_size = min(20, len(data))
            moving_avg = [np.mean(data[max(0, i-window_size):i+1]) for i in range(len(data))]
            plt.plot(epochs, moving_avg, 'r-', linewidth=2, label=f'Moving Avg (window={window_size})')
            plt.legend()
        
        # 为成功率添加50%基准线
        if name == 'success_rate':
            plt.axhline(y=50, color='r', linestyle='--', alpha=0.5)
            plt.ylim(0, 105)
        
        plt.title(f'Training Curve: {title}', fontsize=14, fontweight='bold')
        plt.xlabel('Episodes', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{save_path}/{name}_curve.png", dpi=200)
        plt.close()
    
    # 绘制组合图表：成功率vs奖励
    if success_rates and rewards:
        plt.figure(figsize=(12, 6))
        plt.title('Success Rate vs Reward', fontsize=14, fontweight='bold')
        plt.plot(epochs, success_rates, 'g-', linewidth=2, label='Success Rate')
        plt.ylabel('Success Rate (%)', fontsize=12)
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        ax2 = plt.twinx()
        ax2.plot(epochs, rewards, 'b-', linewidth=2, alpha=0.6, label='Reward')
        ax2.set_ylabel('Reward Value', fontsize=12)
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/success_vs_reward.png", dpi=200)
        plt.close()
    
    # 创建训练概览图表（4合1）
    if all(len(x) > 0 for x in [rewards, success_rates, epsilons, losses]):
        fig, axs = plt.subplots(2, 2, figsize=(18, 10))
        fig.suptitle('Dueling DQN Training Overview', fontsize=16, fontweight='bold')
        
        # 奖励
        axs[0, 0].plot(epochs, rewards, 'b-', linewidth=2)
        axs[0, 0].set_title('Average Reward', fontsize=12)
        axs[0, 0].set_xlabel('Episodes')
        axs[0, 0].set_ylabel('Reward Value')
        axs[0, 0].grid(True, linestyle='--', alpha=0.7)
        
        # 成功率
        axs[0, 1].plot(epochs, success_rates, 'g-', linewidth=2)
        axs[0, 1].set_title('Success Rate', fontsize=12)
        axs[0, 1].set_xlabel('Episodes')
        axs[0, 1].set_ylabel('Success Rate (%)')
        axs[0, 1].set_ylim(0, 105)
        axs[0, 1].axhline(y=50, color='r', linestyle='--', alpha=0.5)
        axs[0, 1].grid(True, linestyle='--', alpha=0.7)
        
        # Epsilon衰减
        axs[1, 0].plot(epochs, epsilons, 'r-', linewidth=2)
        axs[1, 0].set_title('Epsilon Decay', fontsize=12)
        axs[1, 0].set_xlabel('Episodes')
        axs[1, 0].set_ylabel('Epsilon Value')
        axs[1, 0].grid(True, linestyle='--', alpha=0.7)
        
        # 损失
        axs[1, 1].plot(epochs, losses, 'y-', linewidth=2)
        axs[1, 1].set_title('Training Loss', fontsize=12)
        axs[1, 1].set_xlabel('Episodes')
        axs[1, 1].set_ylabel('Loss Value')
        axs[1, 1].grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"{save_path}/training_overview.png", dpi=200)
        plt.close()

# 自定义训练函数
def custom_train_dueling_dqn(gridworld_class, epochs=3000, mode='random', hidden_size=256,
                     lr=5e-4, batch_size=128, memory_size=10000, use_tensorboard=False,
                     wall_penalty=-5.0, log_interval=100, plot_interval=100):
    """自定义训练函数，每隔log_interval轮次记录结果"""
    from dueling_dqn_lightning import DuelingDQNLightning, action_set, test_model
    
    # 创建必要的目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # 创建时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/training_log_{timestamp}.csv"
    
    # 初始化模型
    model = DuelingDQNLightning(
        input_size=64,
        hidden_size=hidden_size,
        output_size=4,
        learning_rate=lr,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        total_epochs=epochs,
        memory_size=memory_size,
        batch_size=batch_size,
        sync_rate=5
    )
    
    # 创建优化器和调度器
    optimizer = model.get_optimizer()
    scheduler = model.get_scheduler(optimizer)
    
    # 移动方向向量（用于墙体检测）
    move_pos = [(-1,0), (1,0), (0,-1), (0,1)]  # u,d,l,r
    
    # 训练监控变量
    log_epochs = []
    log_rewards = []
    log_lengths = []
    log_success_rates = []
    log_epsilons = []
    log_losses = []
    
    max_steps_per_episode = 100  # 每个episode的最大步数限制
    
    print(f"开始训练，总共 {epochs} 轮...")
    print("=" * 70)
    print(f"每 {log_interval} 轮记录一次性能指标")
    print(f"{'回合':^8} | {'奖励':^8} | {'长度':^8} | {'成功率':^8} | {'Epsilon':^8} | {'损失':^12}")
    print("=" * 70)
    
    # 主训练循环
    for epoch in range(epochs):
        # 创建游戏环境
        game = gridworld_class(size=4, mode=mode)
        state = torch.from_numpy(game.board.render_np().reshape(1,64) + np.random.rand(1,64)/100.0).float()
        
        done = False
        total_reward = 0
        episode_length = 0
        step_loss = 0
        
        # 单个episode循环
        while not done and episode_length < max_steps_per_episode:
            # 获取动作
            action_idx = model.get_action(state)
            
            # 检查是否撞墙
            hit_wall = game.validateMove('Player', move_pos[action_idx]) == 1
            
            # 执行动作
            action = action_set[action_idx]
            game.makeMove(action)
            
            # 获取新状态和奖励
            next_state = torch.from_numpy(game.board.render_np().reshape(1,64) + np.random.rand(1,64)/100.0).float()
            
            # 改进奖励设计
            raw_reward = game.reward()
            if hit_wall:
                reward = wall_penalty  # 撞墙惩罚
            elif raw_reward > 0:
                # 给正奖励加上额外的奖励以鼓励找到目标
                reward = raw_reward + 5.0
            elif raw_reward < 0 and raw_reward != -1:
                # 给负奖励保持原样，鼓励避开陷阱
                reward = raw_reward
            else:
                # 每走一步都有小惩罚，鼓励尽快找到目标
                reward = -0.1
                
            game_over = raw_reward != -1  # 游戏是否结束
            
            # 存储经验
            model.add_experience(state, action_idx, reward, next_state, game_over)
            
            # 如果有足够的样本，进行训练
            if len(model.memory) >= batch_size:
                # 从记忆中随机采样
                minibatch = random.sample(model.memory, batch_size)
                
                # 处理批次数据
                state_batch = torch.cat([s for s, _, _, _, _ in minibatch])
                action_batch = torch.cat([a for _, a, _, _, _ in minibatch])
                reward_batch = torch.cat([r for _, _, r, _, _ in minibatch])
                next_state_batch = torch.cat([ns for _, _, _, ns, _ in minibatch])
                done_batch = torch.cat([d for _, _, _, _, d in minibatch])
                
                # 训练步骤
                batch_data = (state_batch, action_batch, reward_batch, next_state_batch, done_batch)
                loss = model.training_step(batch_data, 0)
                
                # 梯度优化
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), model.gradient_clip)
                optimizer.step()
                
                # 更新学习率调度器
                scheduler.step(loss.item())
                
                step_loss = loss.item()
            
            # 更新状态
            state = next_state
            total_reward += reward
            episode_length += 1
            
            # 检查游戏是否结束
            done = game_over
        
        # 记录本次episode的结果
        model.episode_rewards.append(total_reward)
        model.episode_lengths.append(episode_length)
        
        # 更新ε值
        epsilon = model.update_epsilon()
        
        # 每隔log_interval评估模型并记录
        if epoch % log_interval == 0:
            # 评估模型
            success_rate = test_model(model, gridworld_class, mode=mode, test_episodes=20)
            
            # 记录日志
            log_epochs.append(epoch)
            log_rewards.append(total_reward)
            log_lengths.append(episode_length)
            log_success_rates.append(success_rate)
            log_epsilons.append(epsilon)
            log_losses.append(step_loss)
            
            # 打印进度
            print(f"{epoch:^8d} | {total_reward:^8.1f} | {episode_length:^8d} | "
                  f"{success_rate:^8.1f} | {epsilon:^8.3f} | {step_loss:^12.5f}")
            
            # 保存日志到CSV
            save_training_log(
                log_epochs, log_rewards, log_lengths, 
                log_success_rates, log_epsilons, log_losses, 
                filename=log_file
            )
            
            # 实时绘制训练曲线
            if epoch % plot_interval == 0:
                plot_live_training(
                    log_epochs, log_rewards, log_lengths,
                    log_success_rates, log_epsilons, log_losses
                )
            
            # 每100个epoch保存模型
            if epoch % 100 == 0 and epoch > 0:
                torch.save(model.state_dict(), f'models/dueling_dqn_epoch_{epoch}.pth')
        
        # 可选：每100轮显示简单进度提示
        elif epoch % 100 == 0:
            print(f"训练中: {epoch}/{epochs} 轮完成 ({epoch/epochs*100:.1f}%)")
    
    # 训练结束后保存最终模型
    torch.save(model.state_dict(), 'models/dueling_dqn_final.pth')
    
    # 绘制最终训练曲线
    print("\n绘制最终训练曲线...")
    plot_live_training(
        log_epochs, log_rewards, log_lengths,
        log_success_rates, log_epsilons, log_losses,
        save_path="plots/final_training"
    )
    
    return model

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Dueling DQN训练脚本')
    parser.add_argument('--epochs', type=int, default=5000, help='训练轮数')
    parser.add_argument('--mode', type=str, default='random', help='游戏模式: random或static')
    parser.add_argument('--hidden_size', type=int, default=512, help='隐藏层大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--batch_size', type=int, default=256, help='批次大小')
    parser.add_argument('--memory_size', type=int, default=20000, help='经验回放缓冲区大小')
    parser.add_argument('--wall_penalty', type=float, default=-5.0, help='撞墙惩罚')
    parser.add_argument('--log_interval', type=int, default=100, help='日志记录间隔')
    parser.add_argument('--plot_interval', type=int, default=100, help='绘图间隔')
    args = parser.parse_args()
    
    # 设置种子以确保结果可复现
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 创建配置
    config = {
        "epochs": args.epochs,
        "mode": args.mode,
        "hidden_size": args.hidden_size,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "memory_size": args.memory_size,
        "wall_penalty": args.wall_penalty,
        "log_interval": args.log_interval,
        "plot_interval": args.plot_interval
    }
    
    print("=" * 50)
    print("Dueling DQN with PyTorch Lightning 架构")
    print("=" * 50)
    print(f"配置：")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("=" * 50)
    
    # 开始训练
    print("\n开始训练...")
    model = custom_train_dueling_dqn(
        Gridworld,
        **config
    )
    
    # 进行最终评估
    print("\n开始最终评估...")
    success_rate = test_model(
        model, 
        Gridworld, 
        mode=config["mode"], 
        test_episodes=100, 
        visualize=True
    )
    
    print("\n" + "=" * 50)
    print(f"最终成功率: {success_rate:.1f}%")
    print("=" * 50) 