#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dueling DQN 训练恢复脚本
用于从已有的checkpoint继续训练模型
"""

import os
import torch
import numpy as np
from dueling_dqn_lightning import train_dueling_dqn, test_model, DuelingDQNLightning
from Gridworld import Gridworld
import random
import argparse

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='恢复Dueling DQN训练')
    parser.add_argument('--checkpoint', type=str, default='models/dueling_dqn_final.pth',
                        help='要恢复的模型checkpoint路径')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='继续训练的轮数')
    parser.add_argument('--mode', type=str, default='random',
                        help='游戏模式: random或者static')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='隐藏层大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='批次大小')
    parser.add_argument('--memory_size', type=int, default=20000,
                        help='经验回放缓冲区大小')
    parser.add_argument('--wall_penalty', type=float, default=-5.0,
                        help='撞墙惩罚')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='初始epsilon值')
    parser.add_argument('--epsilon_min', type=float, default=0.01,
                        help='最小epsilon值')
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
        "use_tensorboard": False,
        "wall_penalty": args.wall_penalty
    }
    
    print("=" * 50)
    print("恢复 Dueling DQN 训练")
    print("=" * 50)
    print(f"checkpoint: {args.checkpoint}")
    print(f"配置：")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("=" * 50)
    
    # 创建模型
    model = DuelingDQNLightning(
        input_size=64,
        hidden_size=config["hidden_size"],
        output_size=4,
        learning_rate=config["lr"],
        gamma=0.99,
        epsilon=args.epsilon,           # 使用较低的探索率继续训练
        epsilon_min=args.epsilon_min,
        total_epochs=config["epochs"],
        memory_size=config["memory_size"],
        batch_size=config["batch_size"],
        sync_rate=5
    )
    
    # 加载checkpoint
    if os.path.exists(args.checkpoint):
        print(f"加载checkpoint: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint))
    else:
        print(f"警告: 找不到checkpoint {args.checkpoint}，将从头开始训练")
    
    # 创建优化器和调度器
    optimizer = model.get_optimizer()
    scheduler = model.get_scheduler(optimizer)
    
    # 移动方向向量（用于墙体检测）
    move_pos = [(-1,0), (1,0), (0,-1), (0,1)]  # u,d,l,r
    
    # 训练监控变量
    log_epochs = []
    log_loss = []
    log_success = []
    log_epsilon = []
    log_avg_reward = []
    
    max_steps_per_episode = 100  # 每个episode的最大步数限制
    
    # 开始训练
    print("\n开始训练...")
    
    for epoch in range(config["epochs"]):
        # 创建游戏环境
        game = Gridworld(size=4, mode=config["mode"])
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
            from dueling_dqn_lightning import action_set
            action = action_set[action_idx]
            game.makeMove(action)
            
            # 获取新状态和奖励
            next_state = torch.from_numpy(game.board.render_np().reshape(1,64) + np.random.rand(1,64)/100.0).float()
            
            # 改进奖励设计
            raw_reward = game.reward()
            if hit_wall:
                reward = config["wall_penalty"]  # 撞墙惩罚
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
            if len(model.memory) >= config["batch_size"]:
                # 从记忆中随机采样
                minibatch = random.sample(model.memory, config["batch_size"])
                
                # 处理批次数据
                state_batch = torch.cat([s for s, _, _, _, _ in minibatch])
                action_batch = torch.cat([a for _, a, _, _, _ in minibatch])
                reward_batch = torch.cat([r for _, _, r, _, _ in minibatch])
                next_state_batch = torch.cat([ns for _, _, _, ns, _ in minibatch])
                done_batch = torch.cat([d for _, _, _, _, d in minibatch])
                
                # 训练步骤
                batch_data = (state_batch, action_batch, reward_batch, next_state_batch, done_batch)
                loss = model.training_step(batch_data, 0)
                
                # 梯度优化 - 使用标准的PyTorch API
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
        model.update_epsilon()
        
        # 每隔一定间隔评估模型
        if epoch % 10 == 0:
            success_rate = test_model(model, Gridworld, mode=config["mode"], test_episodes=20)
            model.success_rates.append(success_rate)
            
            # 计算过去10个episode的平均奖励
            avg_reward = np.mean(model.episode_rewards[-10:]) if len(model.episode_rewards) >= 10 else np.mean(model.episode_rewards)
            
            print(f"Episode {epoch}, Reward: {total_reward:.1f}, Length: {episode_length}, "
                  f"Success: {success_rate:.1f}%, Epsilon: {model.epsilon:.3f}, "
                  f"Loss: {step_loss:.5f}")
            
            # 记录指标
            log_epochs.append(epoch)
            log_loss.append(step_loss)
            log_success.append(success_rate)
            log_epsilon.append(model.epsilon)
            log_avg_reward.append(avg_reward)
            
            # 每100个epoch保存模型
            if epoch % 100 == 0 and epoch > 0:
                torch.save(model.state_dict(), f'models/dueling_dqn_resume_epoch_{epoch}.pth')
    
    # 训练结束后保存最终模型
    torch.save(model.state_dict(), 'models/dueling_dqn_resume_final.pth')
    
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

    # 使用這個繼續訓練 python resume_training.py --checkpoint models/dueling_dqn_final.pth --epochs 1000 