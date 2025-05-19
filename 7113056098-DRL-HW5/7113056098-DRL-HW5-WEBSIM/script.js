let gridSize = 5;
let startSet = false;
let endSet = false;
let obstacles = 0;
let maxObstacles = 0;
let start_index = -1;
let end_index = -1;
let obstacle_indices = [];

function createGridCells(container, size, content = null, classes = []) {
    container.innerHTML = "";
    container.style.gridTemplateColumns = `repeat(${size}, 50px)`;

    for (let i = 0; i < size * size; i++) {
        const cell = document.createElement("div");
        cell.classList.add("cell");
        classes.forEach(cls => cell.classList.add(cls));
        cell.dataset.index = i;

        if (content && content[i] !== undefined) {
            cell.textContent = content[i];
        } else {
             cell.textContent = i; // Default content for input grid
        }

        if (container.id === 'grid-container') {
            if (i === start_index) cell.classList.add('start');
            if (i === end_index) cell.classList.add('end');
            if (obstacle_indices.includes(i)) cell.classList.add('obstacle');
            cell.addEventListener("click", () => handleCellClick(cell));
        }

        container.appendChild(cell);
    }
}

function generateGrid() {
    gridSize = parseInt(document.getElementById("gridSize").value);
    if (isNaN(gridSize) || gridSize < 5 || gridSize > 9) {
        alert("Please enter a number between 5 and 9.");
        return;
    }

    maxObstacles = gridSize - 2;
    startSet = false;
    endSet = false;
    obstacles = 0;
    start_index = -1;
    end_index = -1;
    obstacle_indices = [];

    const inputContainer = document.getElementById("grid-container");
    createGridCells(inputContainer, gridSize);

    document.getElementById("random-policy-grid").innerHTML = "";
    document.getElementById("random-value-grid").innerHTML = "";
    document.getElementById("optimal-policy-grid").innerHTML = "";
    document.getElementById("optimal-value-grid").innerHTML = "";
}

function handleCellClick(cell) {
    const index = parseInt(cell.dataset.index);
    if (!startSet) {
        const currentStart = document.querySelector('.start');
        if (currentStart) currentStart.classList.remove('start');
        cell.classList.add("start");
        startSet = true;
        start_index = index;
    } else if (!endSet && !cell.classList.contains("start")) {
        const currentEnd = document.querySelector('.end');
        if (currentEnd) currentEnd.classList.remove('end');
        cell.classList.add("end");
        endSet = true;
        end_index = index;
    } else if (obstacles < maxObstacles && !cell.classList.contains("start") && !cell.classList.contains("end") && !cell.classList.contains("obstacle")) {
        cell.classList.add("obstacle");
        obstacle_indices.push(index);
        obstacles++;
    } else if (cell.classList.contains("obstacle")) {
        cell.classList.remove("obstacle");
        obstacle_indices = obstacle_indices.filter(i => i !== index);
        obstacles--;
    }
}

function displayValueFunctionInContainer(containerId, valueFunction, gridSize) {
    const container = document.getElementById(containerId);
    const values = valueFunction.map(value => value.toFixed(2));
    createGridCells(container, gridSize, values, ['value-cell']);
}

function displayPolicyInContainer(containerId, policy, gridSize) {
    const container = document.getElementById(containerId);
    const actionMap = {
        'up': '↑',
        'down': '↓',
        'left': '←',
        'right': '→',
        'terminal': 'X'
    };
    const policyContent = [];
    for(let i = 0; i < gridSize * gridSize; i++) {
        policyContent.push(actionMap[policy[i]] || '');
    }
    createGridCells(container, gridSize, policyContent, ['policy-cell']);
}

function generateRandomPolicy() {
    if (start_index === -1 || end_index === -1) {
        alert("Please set start and end points first.");
        return;
    }

    const policy = {};
    const actions = ['up', 'down', 'left', 'right'];
    for (let i = 0; i < gridSize * gridSize; i++) {
        if (i === end_index || obstacle_indices.includes(i)) {
            policy[i] = 'terminal';
        } else {
            policy[i] = actions[Math.floor(Math.random() * actions.length)];
        }
    }

    const valueFunction = Array.from({length: gridSize * gridSize}, () => parseFloat((Math.random() * 5 - 2).toFixed(2)));

    displayPolicyInContainer("random-policy-grid", policy, gridSize);
    displayValueFunctionInContainer("random-value-grid", valueFunction, gridSize);
}

// --- 這裡是重點：價值迭代與策略推導都在 JS ---
function rewardFunction(state, endState, obstacleStates) {
    if (state[0] === endState[0] && state[1] === endState[1]) return 1;
    for (let obs of obstacleStates) {
        if (state[0] === obs[0] && state[1] === obs[1]) return -1;
    }
    return -0.1;
}

function valueIteration(gridSize, endState, obstacleStates, gamma=0.9) {
    let V = Array.from({length: gridSize}, () => Array(gridSize).fill(0));
    for (let iter = 0; iter < 100; iter++) {
        let V_old = V.map(row => row.slice());
        for (let r = 0; r < gridSize; r++) {
            for (let c = 0; c < gridSize; c++) {
                let state = [r, c];
                if ((r === endState[0] && c === endState[1]) || obstacleStates.some(obs => obs[0] === r && obs[1] === c)) continue;
                let actionValues = [];
                for (let [dr, dc] of [[-1,0],[1,0],[0,-1],[0,1]]) {
                    let next_r = r + dr, next_c = c + dc;
                    if (0 <= next_r && next_r < gridSize && 0 <= next_c && next_c < gridSize) {
                        let reward = rewardFunction([next_r, next_c], endState, obstacleStates);
                        actionValues.push(reward + gamma * V_old[next_r][next_c]);
                    } else {
                        actionValues.push(-100);
                    }
                }
                if (actionValues.length) V[r][c] = Math.max(...actionValues);
            }
        }
    }
    return V;
}

function extractPolicy(V, gridSize, endState, obstacleStates, gamma=0.9) {
    const actions = {
        'up': [-1, 0],
        'down': [1, 0],
        'left': [0, -1],
        'right': [0, 1]
    };
    let policy = {};
    for (let r = 0; r < gridSize; r++) {
        for (let c = 0; c < gridSize; c++) {
            let state = [r, c];
            let idx = r * gridSize + c;
            if ((r === endState[0] && c === endState[1]) || obstacleStates.some(obs => obs[0] === r && obs[1] === c)) {
                policy[idx] = 'terminal';
                continue;
            }
            let bestAction = null;
            let bestValue = -Infinity;
            for (let actionName in actions) {
                let [dr, dc] = actions[actionName];
                let next_r = r + dr, next_c = c + dc;
                if (0 <= next_r && next_r < gridSize && 0 <= next_c && next_c < gridSize) {
                    let reward = rewardFunction([next_r, next_c], endState, obstacleStates);
                    let value = reward + gamma * V[next_r][next_c];
                    if (value > bestValue) {
                        bestValue = value;
                        bestAction = actionName;
                    }
                }
            }
            policy[idx] = bestAction;
        }
    }
    return policy;
}

function computePolicy() {
    if (start_index === -1 || end_index === -1) {
        alert("Please set start and end points first.");
        return;
    }
    const indexToCoords = (index, size) => [Math.floor(index / size), index % size];
    const startState = indexToCoords(start_index, gridSize);
    const endState = indexToCoords(end_index, gridSize);
    const obstacleStates = obstacle_indices.map(i => indexToCoords(i, gridSize));

    // 執行價值迭代
    const V = valueIteration(gridSize, endState, obstacleStates);
    // 推導最佳策略
    const policy = extractPolicy(V, gridSize, endState, obstacleStates);

    // 展平成一維
    const valueFunctionData = V.flat();

    displayValueFunctionInContainer("optimal-value-grid", valueFunctionData, gridSize);
    displayPolicyInContainer("optimal-policy-grid", policy, gridSize);
}

// 頁面載入時初始化
document.addEventListener('DOMContentLoaded', () => {
    generateGrid();
});