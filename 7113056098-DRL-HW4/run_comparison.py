import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import copy
from Gridworld import Gridworld

# 從第3章程式_ALL_IN_ONE.py導入所需的類和函數
from ch3_ALL_IN_ONE import DoubleDQN, DuelingDQN, train_double_dqn, train_dueling_dqn, test_model, action_set

def run_experiment():
    print("開始訓練 Double DQN...")
    double_dqn_model, double_losses = train_double_dqn(epochs=1000)
    
    print("\n開始訓練 Dueling DQN...")
    dueling_dqn_model, dueling_losses = train_dueling_dqn(epochs=1000)
    
    # 繪製訓練損失對比圖
    plt.figure(figsize=(12,6))
    plt.plot(double_losses, label='Double DQN')
    plt.plot(dueling_losses, label='Dueling DQN')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.title('Double DQN vs Dueling DQN Training Loss Comparison')
    plt.legend()
    plt.savefig('dqn_comparison.png')
    plt.close()
    
    # 測試模型性能
    print("\n測試 Double DQN 性能:")
    double_wins = 0
    for i in range(100):
        if test_model(double_dqn_model, mode='player', display=False):
            double_wins += 1
    print(f"Double DQN 勝率: {double_wins}%")
    
    print("\n測試 Dueling DQN 性能:")
    dueling_wins = 0
    for i in range(100):
        if test_model(dueling_dqn_model, mode='player', display=False):
            dueling_wins += 1
    print(f"Dueling DQN 勝率: {dueling_wins}%")
    
    # 保存結果
    with open('comparison_results.txt', 'w') as f:
        f.write(f"Double DQN 勝率: {double_wins}%\n")
        f.write(f"Dueling DQN 勝率: {dueling_wins}%\n")

if __name__ == "__main__":
    run_experiment() 