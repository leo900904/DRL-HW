from flask import Flask, render_template, request, jsonify
import numpy as np

# 假設格子大小由前端傳遞
grid_size = 5  # 預設值，稍後從前端獲取

# 定義動作空間
actions = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
}

# 獎勵函數
def reward_function(state, end_state, obstacle_states):
    if state == end_state:
        return 1
    elif state in obstacle_states:
        return -1
    else:
        return -0.1

# 價值迭代算法
def value_iteration(grid_size, end_state, obstacle_states, gamma=0.9):
    # 初始化價值函數
    V = np.zeros((grid_size, grid_size))
    # 迭代次數
    iterations = 100  # 可以調整
    for _ in range(iterations):
        V_old = np.copy(V)
        for r in range(grid_size):
            for c in range(grid_size):
                state = (r, c)
                if state == end_state or state in obstacle_states:
                    continue
                # 計算每個動作的價值
                action_values = []
                for action_name, (dr, dc) in actions.items():
                    next_r, next_c = r + dr, c + dc
                    # 檢查是否超出邊界
                    if 0 <= next_r < grid_size and 0 <= next_c < grid_size:
                        next_state = (next_r, next_c)
                        reward = reward_function(next_state, end_state, obstacle_states)
                        action_values.append(reward + gamma * V_old[next_r, next_c])
                    else:
                        # 邊界外的狀態，給予較低的價值
                        action_values.append(-100)  # 或者其他懲罰值
                # 選擇最佳動作
                if action_values:
                    V[r, c] = np.max(action_values)
    return V

# 推導最佳策略
def extract_policy(V, grid_size, end_state, obstacle_states, gamma=0.9):
    policy = {}
    for r in range(grid_size):
        for c in range(grid_size):
            state = (r, c)
            if state == end_state or state in obstacle_states:
                policy[state] = 'terminal'
                continue
            best_action = None
            best_value = -np.inf
            for action_name, (dr, dc) in actions.items():
                next_r, next_c = r + dr, c + dc
                if 0 <= next_r < grid_size and 0 <= next_c < grid_size:
                    next_state = (next_r, next_c)
                    reward = reward_function(next_state, end_state, obstacle_states)
                    value = reward + gamma * V[next_r, next_c]
                    if value > best_value:
                        best_value = value
                        best_action = action_name
            policy[state] = best_action
    return policy

app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/update_grid', methods=['POST'])
def update_grid():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        grid_size = int(data.get('gridSize', 5))
        if grid_size < 5 or grid_size > 9:
            return jsonify({"error": "Grid size must be between 5 and 9"}), 400

        start_index = int(data.get('startIndex', -1))
        end_index = int(data.get('endIndex', -1))
        obstacle_indices = [int(i) for i in data.get('obstacleIndices', [])]

        if start_index == -1 or end_index == -1:
            return jsonify({"error": "Start and end points must be set"}), 400

        # 將索引轉換為 (row, col) 坐標
        def index_to_coords(index, size):
            return (index // size, index % size)

        start_state = index_to_coords(start_index, grid_size)
        end_state = index_to_coords(end_index, grid_size)
        obstacle_states = [index_to_coords(i, grid_size) for i in obstacle_indices]

        # 執行價值迭代
        V = value_iteration(grid_size, end_state, obstacle_states)
        policy = extract_policy(V, grid_size, end_state, obstacle_states)

        # 將價值函數和策略轉換為前端可用的格式
        value_function_data = V.flatten().tolist()
        policy_data = {}
        for r in range(grid_size):
            for c in range(grid_size):
                state = (r, c)
                index = r * grid_size + c
                if state in policy:
                    policy_data[index] = policy[state]
                else:
                    policy_data[index] = 'terminal'

        return jsonify({
            "message": "Grid updated successfully!",
            "valueFunction": value_function_data,
            "policy": policy_data,
            "gridSize": grid_size
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
