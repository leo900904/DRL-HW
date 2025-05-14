#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dueling DQN Agent 可视化工具
用于展示训练后模型在Gridworld环境中的表现
"""

import os
import torch
import time
import argparse
import numpy as np
from dueling_dqn_lightning import DuelingDQNLightning, test_model
from Gridworld import Gridworld

def visualize_agent(model_path, episodes=5, mode='random', delay=0.5):
    """
    可视化智能体在Gridworld环境中的表现
    
    Args:
        model_path: 模型权重文件路径
        episodes: 要运行的episode数量
        mode: Gridworld环境模式 ('static', 'random', 'player')
        delay: 每步之间的延迟时间(秒)
    """
    # 初始化模型
    model = DuelingDQNLightning()
    
    # 加载模型权重
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"成功加载模型: {model_path}")
    else:
        print(f"模型文件不存在: {model_path}")
        return
    
    # 将模型设为评估模式
    model.eval()
    
    # 记录成功次数
    wins = 0
    
    # 运行指定次数的episode
    for episode in range(episodes):
        print(f"\n=== Episode {episode+1}/{episodes} ===")
        
        # 创建游戏环境
        game = Gridworld(size=4, mode=mode)
        
        # 显示初始状态
        print("\n初始状态:")
        game.display()
        
        # 获取初始状态
        state = torch.from_numpy(game.board.render_np().reshape(1,64) + np.random.rand(1,64)/100.0).float()
        
        # 记录步数和总奖励
        steps = 0
        total_reward = 0
        done = False
        
        # 单个episode循环
        while not done and steps < 50:  # 最多50步
            # 获取动作
            with torch.no_grad():
                q_values = model(state)
                action_idx = torch.argmax(q_values).item()
            
            # 将动作索引转换为字母
            action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
            action = action_set[action_idx]
            
            # 显示当前动作
            print(f"\n步骤 {steps+1}: 执行动作 '{action}'")
            
            # 执行动作
            game.makeMove(action)
            
            # 显示执行动作后的状态
            game.display()
            
            # 获取新状态和奖励
            state = torch.from_numpy(game.board.render_np().reshape(1,64) + np.random.rand(1,64)/100.0).float()
            reward = game.reward()
            total_reward += reward
            
            # 检查游戏是否结束
            done = reward != -1
            
            # 输出当前步骤信息
            print(f"奖励: {reward}")
            
            # 检查是否胜利
            if reward == 10:
                print("\n🎉 成功! 智能体找到了目标!")
                wins += 1
                break
            elif reward == -10:
                print("\n❌ 失败! 智能体踩到了陷阱!")
                break
            
            # 步数加1
            steps += 1
            
            # 延迟一段时间以便观察
            time.sleep(delay)
        
        if not done:
            print("\n⚠️ 达到最大步数限制!")
        
        print(f"\nEpisode {episode+1} 总结:")
        print(f"总步数: {steps}")
        print(f"总奖励: {total_reward}")
        
        # 在episodes之间短暂暂停
        if episode < episodes - 1:
            print("\n3秒后开始下一个episode...")
            time.sleep(3)
    
    # 显示总体表现
    print(f"\n=== 总体表现 ({episodes} episodes) ===")
    print(f"成功率: {wins/episodes*100:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dueling DQN Agent可视化工具')
    parser.add_argument('--model', type=str, default='models/dueling_dqn_final.pth',
                        help='模型文件路径')
    parser.add_argument('--episodes', type=int, default=5,
                        help='要运行的episode数量')
    parser.add_argument('--mode', type=str, default='random',
                        choices=['static', 'random', 'player'],
                        help='Gridworld环境模式')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='每步之间的延迟时间(秒)')
    
    args = parser.parse_args()
    
    visualize_agent(args.model, args.episodes, args.mode, args.delay) 