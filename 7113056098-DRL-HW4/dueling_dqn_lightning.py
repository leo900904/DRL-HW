import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
from collections import deque
import random
import os
from datetime import datetime

# 将动作的字母与数字对应起来
action_set = {
    0: 'u',  # "0"代表"向上"
    1: 'd',  # "1"代表"向下"
    2: 'l',  # "2"代表"向左"
    3: 'r'   # "3"代表"向右"
}

# Dueling DQN 架构的 Lightning 模块
class DuelingDQNLightning(pl.LightningModule):
    def __init__(self, input_size=64, hidden_size=150, output_size=4, 
                 learning_rate=1e-3, gamma=0.99, epsilon=1.0, epsilon_min=0.01, 
                 epsilon_decay=None, total_epochs=3000, sync_rate=5, 
                 memory_size=5000, batch_size=256, gradient_clip=1.0):
        super().__init__()
        self.save_hyperparameters()
        
        # 特征提取网络 - 增加更多层
        self.feature_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # 价值流 - 增加更多单元
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, 1)
        )
        
        # 优势流 - 增加更多单元
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, output_size)
        )
        
        # 目标网络（Dueling结构）- 增加更多层
        self.target_feature_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        self.target_value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, 1)
        )
        
        self.target_advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, output_size)
        )
        
        # 初始化目标网络权重
        self._sync_target_network()
        
        # 经验回放缓冲区
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        
        # 训练参数
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.total_epochs = total_epochs
        self.sync_rate = sync_rate  # 目标网络同步频率
        self.gradient_clip = gradient_clip  # 梯度裁剪值
        
        # 线性衰减
        self.epsilon_decay = (epsilon - epsilon_min) / total_epochs if epsilon_decay is None else epsilon_decay
        
        # 指标追踪
        self.training_step_outputs = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        self.learning_rate = learning_rate
        
        # 训练步数计数器
        self.training_steps = 0
    
    def forward(self, x):
        features = self.feature_network(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        # Dueling DQN的Q值计算：Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        return values + (advantages - advantages.mean(dim=1, keepdim=True))
    
    def target_forward(self, x):
        features = self.target_feature_network(x)
        values = self.target_value_stream(features)
        advantages = self.target_advantage_stream(features)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))
    
    def _sync_target_network(self):
        # 将当前网络的权重同步到目标网络
        self.target_feature_network.load_state_dict(self.feature_network.state_dict())
        self.target_value_stream.load_state_dict(self.value_stream.state_dict())
        self.target_advantage_stream.load_state_dict(self.advantage_stream.state_dict())
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=200, 
            min_lr=1e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "frequency": 100,
                "interval": "step"
            }
        }
    
    # 添加获取优化器的方法
    def get_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    # 添加获取学习率调度器的方法
    def get_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=200, 
            min_lr=1e-5
        )
    
    def training_step(self, batch, batch_idx):
        # 解包批次数据
        states, actions, rewards, next_states, dones = batch
        
        # 使用当前网络计算Q值
        current_q_values = self(states)
        
        # Double DQN：使用当前网络选择动作，使用目标网络估计Q值
        with torch.no_grad():
            next_q_values = self(next_states)
            best_actions = torch.argmax(next_q_values, dim=1)
            next_q_values_target = self.target_forward(next_states)
            max_next_q = next_q_values_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值
        expected_q_values = rewards + (1 - dones.float()) * self.gamma * max_next_q
        
        # 获取所选动作的Q值
        actions_q_values = current_q_values.gather(1, actions.long().unsqueeze(1)).squeeze(1)
        
        # 计算损失，使用Huber损失增强稳定性
        loss = F.smooth_l1_loss(actions_q_values, expected_q_values)
        
        # 记录指标 - 修改为不依赖Trainer的方式
        # self.log('train_loss', loss, prog_bar=True)
        
        self.training_steps += 1
        if self.training_steps % self.sync_rate == 0:
            self._sync_target_network()
        
        return loss
    
    def get_action(self, state, evaluate=False):
        # 用于选择动作的方法
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, 3)  # 探索
        
        with torch.no_grad():
            q_values = self(state)
            return torch.argmax(q_values).item()  # 利用

    def update_epsilon(self):
        # 更新epsilon值（ε-贪心策略）
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
            return max(self.epsilon, self.epsilon_min)
        return self.epsilon

    def add_experience(self, state, action, reward, next_state, done):
        # 添加经验到回放缓冲区
        self.memory.append((state, torch.tensor([action]), 
                           torch.tensor([reward]), next_state, 
                           torch.tensor([done], dtype=torch.float)))

# 训练函数
def train_dueling_dqn(gridworld_class, epochs=3000, mode='random', hidden_size=256,
                     lr=5e-4, batch_size=128, memory_size=10000, use_tensorboard=False,
                     wall_penalty=-5.0, tensorboard_name="dueling_dqn"):
    from Gridworld import Gridworld  # 导入在训练函数内部，以确保可以访问
    
    # 创建必要的目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    if use_tensorboard:
        os.makedirs('logs', exist_ok=True)
        logger = TensorBoardLogger("logs", name=tensorboard_name)
    else:
        logger = None
    
    # 初始化模型和回调
    model = DuelingDQNLightning(
        input_size=64,
        hidden_size=hidden_size,
        output_size=4,
        learning_rate=lr,
        gamma=0.99,  # 更高的折扣因子
        epsilon=1.0,
        epsilon_min=0.01,  # 更低的最小探索率
        total_epochs=epochs,
        memory_size=memory_size,
        batch_size=batch_size,
        sync_rate=5  # 更频繁地同步目标网络
    )
    
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
            success_rate = test_model(model, gridworld_class, mode=mode, test_episodes=20)
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
                torch.save(model.state_dict(), f'models/dueling_dqn_epoch_{epoch}.pth')
    
    # 训练结束后保存最终模型
    torch.save(model.state_dict(), 'models/dueling_dqn_final.pth')
    
    # 绘制训练曲线
    plot_training_curves(log_epochs, log_loss, log_success, log_epsilon, log_avg_reward)
    
    return model

# 测试函数
def test_model(model, gridworld_class, mode='random', test_episodes=100, visualize=False):
    from Gridworld import Gridworld
    wins = 0
    rewards = []
    steps = []
    
    for _ in range(test_episodes):
        game = gridworld_class(size=4, mode=mode)
        state = torch.from_numpy(game.board.render_np().reshape(1,64) + np.random.rand(1,64)/100.0).float()
        done = False
        total_reward = 0
        step_count = 0
        max_steps = 50  # 防止无限循环
        
        if visualize:
            print("\nInitial state:")
            game.display()
        
        while not done and step_count < max_steps:
            # 评估模式下选择动作
            action_idx = model.get_action(state, evaluate=True)
            action = action_set[action_idx]
            
            # 执行动作
            game.makeMove(action)
            
            if visualize:
                print(f"Step {step_count+1}, Action: {action}")
                game.display()
            
            # 更新状态
            state = torch.from_numpy(game.board.render_np().reshape(1,64) + np.random.rand(1,64)/100.0).float()
            reward = game.reward()
            total_reward += reward
            
            # 检查游戏是否结束
            done = reward != -1
            step_count += 1
            
            # 检查是否胜利
            if reward > 0:
                wins += 1
                if visualize:
                    print("Win! 🎉")
                break
        
        rewards.append(total_reward)
        steps.append(step_count)
        
        if visualize and not done:
            print("Max steps reached without result.")
    
    win_rate = 100.0 * wins / test_episodes
    avg_reward = np.mean(rewards)
    avg_steps = np.mean(steps)
    
    if visualize:
        print(f"\nTest Results ({test_episodes} episodes):")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Steps: {avg_steps:.2f}")
    
    return win_rate

# 绘制训练曲线
def plot_training_curves(epochs, losses, success_rates, epsilons, rewards):
    # 设置绘图样式
    plt.style.use('ggplot')
    
    # 创建包含4个子图的图表
    fig, axs = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Dueling DQN with PyTorch Lightning - 训练指标', fontsize=20, fontweight='bold')
    
    # 1. 损失曲线
    axs[0, 0].plot(epochs, losses, 'b-', linewidth=2, label='训练损失')
    # 计算移动平均线以显示趋势
    window_size = min(20, len(losses))
    if window_size > 0:
        moving_avg = [np.mean(losses[max(0, i-window_size):i+1]) for i in range(len(losses))]
        axs[0, 0].plot(epochs, moving_avg, 'r-', linewidth=2, label='移动平均 (窗口={})'.format(window_size))
    axs[0, 0].set_title('训练损失曲线', fontsize=16, fontweight='bold')
    axs[0, 0].set_xlabel('训练轮次', fontsize=14)
    axs[0, 0].set_ylabel('损失值', fontsize=14)
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)
    axs[0, 0].legend(fontsize=12)
    # 标记最低损失点
    if len(losses) > 0:
        min_loss_idx = np.argmin(losses)
        min_loss = losses[min_loss_idx]
        axs[0, 0].scatter([epochs[min_loss_idx]], [min_loss], color='green', s=100, zorder=5)
        axs[0, 0].annotate(f'最低损失: {min_loss:.4f}',
                          xy=(epochs[min_loss_idx], min_loss),
                          xytext=(epochs[min_loss_idx], min_loss*1.5),
                          arrowprops=dict(facecolor='black', shrink=0.05),
                          fontsize=12)
    
    # 2. 成功率曲线
    axs[0, 1].plot(epochs, success_rates, 'g-', linewidth=2, label='成功率')
    # 添加水平线标记关键值
    axs[0, 1].axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50%基准')
    axs[0, 1].set_title('成功率曲线', fontsize=16, fontweight='bold')
    axs[0, 1].set_xlabel('训练轮次', fontsize=14)
    axs[0, 1].set_ylabel('成功率 (%)', fontsize=14)
    axs[0, 1].set_ylim(0, 105)  # 确保y轴从0到100以上，留出标注空间
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)
    axs[0, 1].legend(fontsize=12)
    # 标记最高成功率点
    if len(success_rates) > 0:
        max_success_idx = np.argmax(success_rates)
        max_success = success_rates[max_success_idx]
        axs[0, 1].scatter([epochs[max_success_idx]], [max_success], color='red', s=100, zorder=5)
        axs[0, 1].annotate(f'最高成功率: {max_success:.1f}%',
                          xy=(epochs[max_success_idx], max_success),
                          xytext=(epochs[max_success_idx], max_success - 20),
                          arrowprops=dict(facecolor='black', shrink=0.05),
                          fontsize=12)
    
    # 3. Epsilon衰减曲线
    axs[1, 0].plot(epochs, epsilons, 'r-', linewidth=2, label='Epsilon值')
    axs[1, 0].set_title('Epsilon探索率衰减曲线', fontsize=16, fontweight='bold')
    axs[1, 0].set_xlabel('训练轮次', fontsize=14)
    axs[1, 0].set_ylabel('Epsilon值', fontsize=14)
    axs[1, 0].set_ylim(0, 1.1)  # 确保y轴从0到1.1
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)
    axs[1, 0].legend(fontsize=12)
    
    # 4. 平均奖励曲线
    axs[1, 1].plot(epochs, rewards, 'y-', linewidth=2, label='平均奖励')
    # 计算移动平均线以显示趋势
    if len(rewards) > 0:
        window_size = min(20, len(rewards))
        moving_avg = [np.mean(rewards[max(0, i-window_size):i+1]) for i in range(len(rewards))]
        axs[1, 1].plot(epochs, moving_avg, 'c-', linewidth=2, label='移动平均 (窗口={})'.format(window_size))
    axs[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)  # 零线
    axs[1, 1].set_title('平均奖励曲线', fontsize=16, fontweight='bold')
    axs[1, 1].set_xlabel('训练轮次', fontsize=14)
    axs[1, 1].set_ylabel('平均奖励', fontsize=14)
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)
    axs[1, 1].legend(fontsize=12)
    
    # 添加训练信息标注
    # 在图表底部添加训练参数信息
    fig.text(0.5, 0.01, 
            f'训练参数: 轮次={len(epochs)}, 最终Epsilon={epsilons[-1]:.3f}, 最高成功率={max(success_rates):.1f}%', 
            ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('plots/dueling_dqn_training_metrics.png', dpi=300)
    plt.close()
    
    # 单独保存高分辨率图表
    metrics = {
        'loss': (epochs, losses, '训练损失曲线', '训练轮次', '损失值'),
        'success_rate': (epochs, success_rates, '成功率曲线', '训练轮次', '成功率 (%)'),
        'epsilon': (epochs, epsilons, 'Epsilon探索率衰减曲线', '训练轮次', 'Epsilon值'),
        'reward': (epochs, rewards, '平均奖励曲线', '训练轮次', '平均奖励')
    }
    
    for name, (x, y, title, xlabel, ylabel) in metrics.items():
        plt.figure(figsize=(12, 8))
        plt.title(title, fontsize=18, fontweight='bold')
        plt.plot(x, y, linewidth=2.5)
        
        # 对于损失和奖励，添加移动平均线
        if name in ['loss', 'reward'] and len(y) > 0:
            window_size = min(20, len(y))
            moving_avg = [np.mean(y[max(0, i-window_size):i+1]) for i in range(len(y))]
            plt.plot(x, moving_avg, 'r-', linewidth=2, alpha=0.7, label='移动平均 (窗口={})'.format(window_size))
            plt.legend(fontsize=12)
        
        # 对于成功率图，添加50%基准线和标记最高点
        if name == 'success_rate' and len(y) > 0:
            plt.axhline(y=50, color='r', linestyle='--', alpha=0.5)
            max_idx = np.argmax(y)
            max_val = y[max_idx]
            plt.scatter([x[max_idx]], [max_val], color='red', s=100, zorder=5)
            plt.annotate(f'最高成功率: {max_val:.1f}%',
                         xy=(x[max_idx], max_val),
                         xytext=(x[max_idx], max_val - 20),
                         arrowprops=dict(facecolor='black', shrink=0.05),
                         fontsize=12)
            plt.ylim(0, 105)
        
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'plots/dueling_dqn_{name}.png', dpi=300)
        plt.close()
    
    # 创建额外的组合对比图表
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 1, 1)
    plt.title('成功率 vs. 损失', fontsize=16, fontweight='bold')
    plt.plot(epochs, success_rates, 'g-', linewidth=2, label='成功率')
    plt.ylabel('成功率 (%)', fontsize=14)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    ax2 = plt.twinx()
    ax2.plot(epochs, losses, 'b-', linewidth=2, alpha=0.6, label='损失')
    ax2.set_ylabel('损失值', fontsize=14)
    ax2.legend(loc='upper right', fontsize=12)
    
    plt.subplot(2, 1, 2)
    plt.title('成功率 vs. Epsilon', fontsize=16, fontweight='bold')
    plt.plot(epochs, success_rates, 'g-', linewidth=2, label='成功率')
    plt.xlabel('训练轮次', fontsize=14)
    plt.ylabel('成功率 (%)', fontsize=14)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    ax4 = plt.twinx()
    ax4.plot(epochs, epsilons, 'r-', linewidth=2, alpha=0.6, label='Epsilon')
    ax4.set_ylabel('Epsilon值', fontsize=14)
    ax4.legend(loc='upper right', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('plots/dueling_dqn_combined_metrics.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    from Gridworld import Gridworld
    
    # 主训练循环
    print("开始训练 Dueling DQN 使用 PyTorch Lightning...")
    model = train_dueling_dqn(
        Gridworld,
        epochs=3000, 
        mode='random',
        hidden_size=256,
        lr=3e-4,
        batch_size=128,
        memory_size=10000,
        wall_penalty=-5.0
    )
    
    # 最终测试
    print("\n开始进行最终测试...")
    final_success_rate = test_model(model, Gridworld, mode='random', test_episodes=100, visualize=True)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'results_{timestamp}.txt', 'w') as f:
        f.write(f"最终成功率: {final_success_rate:.1f}%\n")
        f.write(f"最终Epsilon值: {model.epsilon:.3f}\n")
        f.write(f"平均Episode长度: {np.mean(model.episode_lengths):.1f}\n")
        f.write(f"平均Episode奖励: {np.mean(model.episode_rewards):.1f}\n") 