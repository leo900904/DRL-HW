from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/update_grid', methods=['POST'])
def update_grid():
    data = request.json
    return jsonify({"message": "Grid updated successfully!", "data": data})

def value_iteration(grid_size, start_index, end_index, obstacles, gamma=0.99, theta=0.0001):
    """
    根據指定的網格大小 (grid_size)、起點 (start_index)、終點 (end_index)、
    以及障礙物位置 (obstacles)，執行 Value Iteration。
    
    - 每走一步給 -0.04 (step cost)
    - 到達終點給 +1
    - 若撞到障礙物或出界，給 -1 懲罰
    - 折扣因子 gamma=0.99
    
    回傳:
      - policy: 每個狀態 (row, col) 對應的最優行動 (例如 '↑', '↓', '←', '→', 或 'GOAL')
      - V: 每個狀態 (row, col) 的最終價值函數
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

    # 四個方向的行動
    actions = {
        '↑': (-1, 0),
        '↓': (1, 0),
        '←': (0, -1),
        '→': (0, 1),
    }

    def is_valid_cell(r, c):
        """檢查 (r,c) 是否在邊界內且不是障礙物"""
        if r < 0 or r >= grid_size or c < 0 or c >= grid_size:
            return False
        idx = r * grid_size + c
        return (idx not in obstacles)

    while True:
        delta = 0
        newV = V.copy()

        for s in states:
            r, c = s
            current_idx = r * grid_size + c

            # 終點的值可固定為 0
            if current_idx == end_index:
                newV[s] = 0
                continue

            best_value = float('-inf')

            for a_key, (dr, dc) in actions.items():
                nr, nc = r + dr, c + dc
                if is_valid_cell(nr, nc):
                    next_idx = nr * grid_size + nc
                    # 到達終點 +1，否則每步 -0.04
                    reward = 1.0 if next_idx == end_index else -0.04
                    q_sa = reward + gamma * V[(nr, nc)]
                else:
                    # 出界或撞障礙物：-1
                    q_sa = -1

                if q_sa > best_value:
                    best_value = q_sa

            newV[s] = best_value
            delta = max(delta, abs(newV[s] - V[s]))

        V = newV
        if delta < theta:
            break

    # 根據最終的 V(s) 推導最優策略
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
    """
    從前端接收:
      - gridSize
      - startIndex
      - endIndex
      - obstacles (陣列)
    並呼叫 value_iteration()，回傳 policy 與 V(s)
    """
    data = request.json
    grid_size = data['gridSize']
    start_index = data['startIndex']
    end_index = data['endIndex']
    obstacles = data['obstacles']

    policy, value_function = value_iteration(grid_size, start_index, end_index, obstacles)

    # 將 dict 的 key 從 (row, col) 轉成字串，以便 JSON 化
    policy_str_keys = {f"{k[0]},{k[1]}": v for k, v in policy.items()}
    value_str_keys = {f"{k[0]},{k[1]}": val for k, val in value_function.items()}

    return jsonify({
        "policy": policy_str_keys,
        "valueFunction": value_str_keys
    })

if __name__ == '__main__':
    app.run(debug=True)
