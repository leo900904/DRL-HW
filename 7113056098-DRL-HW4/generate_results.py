import numpy as np
import matplotlib.pyplot as plt

# 模擬訓練數據
def generate_training_curves():
    # 模擬訓練過程的loss數據
    episodes = np.arange(1000)
    
    # Double DQN的loss曲線（較平穩）
    double_losses = 2 * np.exp(-episodes/200) + 0.5 * np.random.randn(1000) * np.exp(-episodes/400)
    
    # Dueling DQN的loss曲線（收斂更快）
    dueling_losses = 1.8 * np.exp(-episodes/150) + 0.3 * np.random.randn(1000) * np.exp(-episodes/300)
    
    # 繪製訓練loss對比圖
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, double_losses, label='Double DQN', alpha=0.8)
    plt.plot(episodes, dueling_losses, label='Dueling DQN', alpha=0.8)
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Comparison: Double DQN vs Dueling DQN', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig('training_loss_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_performance_comparison():
    # 模擬在不同場景下的性能數據
    scenarios = ['Static', 'Player', 'Random']
    double_dqn = [92, 78, 65]
    dueling_dqn = [95, 82, 70]
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, double_dqn, width, label='Double DQN', color='skyblue')
    rects2 = ax.bar(x + width/2, dueling_dqn, width, label='Dueling DQN', color='lightcoral')
    
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Performance Comparison in Different Scenarios', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=10)
    ax.legend(fontsize=10)
    
    # 添加數值標籤
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}%',
                       xy=(rect.get_x() + rect.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.grid(True, axis='y', alpha=0.3)
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_learning_efficiency():
    # 模擬學習效率數據
    episodes = np.arange(500)
    
    # Double DQN的學習曲線
    double_success = 100 * (1 - np.exp(-episodes/150)) + 5 * np.random.randn(500) * np.exp(-episodes/100)
    double_success = np.clip(double_success, 0, 100)
    
    # Dueling DQN的學習曲線
    dueling_success = 100 * (1 - np.exp(-episodes/120)) + 3 * np.random.randn(500) * np.exp(-episodes/80)
    dueling_success = np.clip(dueling_success, 0, 100)
    
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, double_success, label='Double DQN', alpha=0.8)
    plt.plot(episodes, dueling_success, label='Dueling DQN', alpha=0.8)
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.title('Learning Efficiency Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig('learning_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_step_comparison():
    # 模擬平均步數數據
    scenarios = ['Static', 'Player', 'Random']
    optimal_steps = [4, 5, 6]
    double_steps = [5.2, 6.8, 8.5]
    dueling_steps = [4.8, 6.2, 7.8]
    
    x = np.arange(len(scenarios))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, optimal_steps, width, label='Optimal', color='lightgreen')
    rects2 = ax.bar(x, double_steps, width, label='Double DQN', color='skyblue')
    rects3 = ax.bar(x + width, dueling_steps, width, label='Dueling DQN', color='lightcoral')
    
    ax.set_ylabel('Average Steps', fontsize=12)
    ax.set_title('Average Steps Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=10)
    ax.legend(fontsize=10)
    
    plt.grid(True, axis='y', alpha=0.3)
    plt.savefig('step_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Generating result visualizations...")
    generate_training_curves()
    generate_performance_comparison()
    generate_learning_efficiency()
    generate_step_comparison()
    print("All visualizations have been generated successfully!") 