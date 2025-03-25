from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/update_grid', methods=['POST'])
def update_grid():
    data = request.json
    return jsonify({"message": "Grid updated successfully!", "data": data})

def value_iteration_animate(grid_size, start_index, end_index, obstacles, gamma=0.99, theta=0.0001, record_every=10):
    """
    進行 Value Iteration，並每 record_every 次迭代時記錄一次 V(s) 的快照，
    最後回傳最終的 policy、最終的 V(s) 以及所有快照 (snapshots)。
    """
    def index_to_rc(idx):
        return divmod(idx, grid_size)

    # 建立狀態空間（排除障礙物）
    states = []
    for idx in range(grid_size * grid_size):
        if idx not in obstacles:
            states.append(index_to_rc(idx))
    
    # 初始化 V(s) = 0
    V = {s: 0.0 for s in states}
    snapshots = []  # 用來記錄 V 的快照

    # 定義行動
    actions = {
        '↑': (-1, 0),
        '↓': (1, 0),
        '←': (0, -1),
        '→': (0, 1),
    }

    def is_valid_cell(r, c):
        if r < 0 or r >= grid_size or c < 0 or c >= grid_size:
            return False
        idx = r * grid_size + c
        return idx not in obstacles

    iteration = 0
    while True:
        delta = 0
        newV = V.copy()
        for s in states:
            r, c = s
            current_idx = r * grid_size + c

            # 若為終點，V(s) 固定為 0
            if current_idx == end_index:
                newV[s] = 0
                continue

            best_value = float('-inf')
            for a_key, (dr, dc) in actions.items():
                nr, nc = r + dr, c + dc
                if is_valid_cell(nr, nc):
                    next_idx = nr * grid_size + nc
                    reward = 1.0 if next_idx == end_index else -0.04
                    q_sa = reward + gamma * V[(nr, nc)]
                else:
                    q_sa = -1
                if q_sa > best_value:
                    best_value = q_sa
            newV[s] = best_value
            delta = max(delta, abs(newV[s] - V[s]))
        V = newV
        iteration += 1
        if iteration % record_every == 0:
            # 記錄快照
            snapshots.append({k: v for k, v in V.items()})
        if delta < theta:
            snapshots.append({k: v for k, v in V.items()})
            break

    # 推導最終的 policy
    policy = {}
    for s in states:
        r, c = s
        current_idx = r * grid_size + c
        if current_idx == end_index:
            policy[s] = "GOAL"
            continue
        best_action = None
        best_value = float('-inf')
        for a_key, (dr, dc) in actions.items():
            nr, nc = r + dr, c + dc
            if is_valid_cell(nr, nc):
                next_idx = nr * grid_size + nc
                reward = 1.0 if next_idx == end_index else -0.04
                q_sa = reward + gamma * V[(nr, nc)]
            else:
                q_sa = -1
            if q_sa > best_value:
                best_value = q_sa
                best_action = a_key
        policy[s] = best_action if best_action else "."
    return policy, V, snapshots

@app.route('/animate_policy', methods=['POST'])
def animate_policy():
    """
    從前端接收:
      - gridSize
      - startIndex
      - endIndex
      - obstacles (陣列)
    並使用 value_iteration_animate() 來回傳最終的 policy、最終的 V(s)
    以及 V(s) 的快照 (snapshots) 用於動畫呈現。
    """
    data = request.json
    grid_size = data['gridSize']
    start_index = data['startIndex']
    end_index = data['endIndex']
    obstacles = data['obstacles']
    
    policy, value_function, snapshots = value_iteration_animate(grid_size, start_index, end_index, obstacles)
    
    # 將 dict 的 key 從 (row, col) 轉成字串以便 JSON 序列化
    snapshots_str = []
    for snap in snapshots:
        snap_str = {f"{k[0]},{k[1]}": v for k, v in snap.items()}
        snapshots_str.append(snap_str)
    policy_str_keys = {f"{k[0]},{k[1]}": v for k, v in policy.items()}
    value_str_keys = {f"{k[0]},{k[1]}": val for k, val in value_function.items()}
    
    return jsonify({
        "policy": policy_str_keys,
        "valueFunction": value_str_keys,
        "snapshots": snapshots_str
    })

# 原有的 compute_policy 保持不變
def value_iteration(grid_size, start_index, end_index, obstacles, gamma=0.99, theta=0.0001):
    def index_to_rc(idx):
        return divmod(idx, grid_size)
    states = []
    for idx in range(grid_size * grid_size):
        if idx not in obstacles:
            states.append(index_to_rc(idx))
    V = {s: 0.0 for s in states}
    actions = {
        '↑': (-1, 0),
        '↓': (1, 0),
        '←': (0, -1),
        '→': (0, 1),
    }
    def is_valid_cell(r, c):
        if r < 0 or r >= grid_size or c < 0 or c >= grid_size:
            return False
        idx = r * grid_size + c
        return idx not in obstacles
    while True:
        delta = 0
        newV = V.copy()
        for s in states:
            r, c = s
            current_idx = r * grid_size + c
            if current_idx == end_index:
                newV[s] = 0
                continue
            best_value = float('-inf')
            for a_key, (dr, dc) in actions.items():
                nr, nc = r + dr, c + dc
                if is_valid_cell(nr, nc):
                    next_idx = nr * grid_size + nc
                    reward = 1.0 if next_idx == end_index else -0.04
                    q_sa = reward + gamma * V[(nr, nc)]
                else:
                    q_sa = -1
                if q_sa > best_value:
                    best_value = q_sa
            newV[s] = best_value
            delta = max(delta, abs(newV[s] - V[s]))
        V = newV
        if delta < theta:
            break
    policy = {}
    for s in states:
        r, c = s
        current_idx = r * grid_size + c
        if current_idx == end_index:
            policy[s] = "GOAL"
            continue
        best_action = None
        best_value = float('-inf')
        for a_key, (dr, dc) in actions.items():
            nr, nc = r + dr, c + dc
            if is_valid_cell(nr, nc):
                next_idx = nr * grid_size + nc
                reward = 1.0 if next_idx == end_index else -0.04
                q_sa = reward + gamma * V[(nr, nc)]
            else:
                q_sa = -1
            if q_sa > best_value:
                best_value = q_sa
                best_action = a_key
        policy[s] = best_action if best_action else "."
    return policy, V

@app.route('/compute_policy', methods=['POST'])
def compute_policy():
    data = request.json
    grid_size = data['gridSize']
    start_index = data['startIndex']
    end_index = data['endIndex']
    obstacles = data['obstacles']
    policy, value_function = value_iteration(grid_size, start_index, end_index, obstacles)
    policy_str_keys = {f"{k[0]},{k[1]}": v for k, v in policy.items()}
    value_str_keys = {f"{k[0]},{k[1]}": val for k, val in value_function.items()}
    return jsonify({
        "policy": policy_str_keys,
        "valueFunction": value_str_keys
    })

if __name__ == '__main__':
    app.run(debug=True)
