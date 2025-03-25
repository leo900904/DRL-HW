let gridSize = 5;
let startSet = false;
let endSet = false;
let startIndex = null;
let endIndex = null;
let obstaclesList = [];
let maxObstacles = 0;

function generateGrid() {
    gridSize = parseInt(document.getElementById("gridSize").value);
    if (isNaN(gridSize) || gridSize < 3 || gridSize > 9) {
        alert("Please enter a number between 3 and 9.");
        return;
    }
    maxObstacles = gridSize - 2;
    startSet = false;
    endSet = false;
    startIndex = null;
    endIndex = null;
    obstaclesList = [];

    // 清空輸入網格（位於 #grid-container 中）
    const container = document.getElementById("grid-container");
    container.innerHTML = "";
    container.style.gridTemplateColumns = `repeat(${gridSize}, 50px)`;

    // 生成輸入網格
    for (let i = 0; i < gridSize * gridSize; i++) {
        const cell = document.createElement("div");
        cell.classList.add("cell");
        cell.textContent = i + 1;
        cell.dataset.index = i;
        cell.addEventListener("click", () => handleCellClick(cell));
        container.appendChild(cell);
    }
}

function handleCellClick(cell) {
    const index = parseInt(cell.dataset.index);
    if (!startSet) {
        cell.classList.add("start");
        startSet = true;
        startIndex = index;
    } else if (!endSet && !cell.classList.contains("start")) {
        cell.classList.add("end");
        endSet = true;
        endIndex = index;
    } else if (obstaclesList.length < maxObstacles &&
               !cell.classList.contains("start") &&
               !cell.classList.contains("end")) {
        cell.classList.add("obstacle");
        obstaclesList.push(index);
    }
}

function computePolicy() {
    if (startIndex === null || endIndex === null) {
        alert("Please set both a start cell and an end cell.");
        return;
    }
    const data = {
        gridSize: gridSize,
        startIndex: startIndex,
        endIndex: endIndex,
        obstacles: obstaclesList
    };
    fetch("/compute_policy", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        const policy = result.policy;
        const valueFunction = result.valueFunction;
        const viContainer = document.getElementById("vi-results-container");
        viContainer.innerHTML = "";

        // 建立一個結果列，橫向排列 Value Function 與 Policy 網格
        const row = document.createElement("div");
        row.classList.add("result-row");

        const valueGridContainer = document.createElement("div");
        valueGridContainer.classList.add("grid");
        valueGridContainer.style.gridTemplateColumns = `repeat(${gridSize}, 50px)`;
        fillValueGrid(valueGridContainer, valueFunction);

        const policyGridContainer = document.createElement("div");
        policyGridContainer.classList.add("grid");
        policyGridContainer.style.gridTemplateColumns = `repeat(${gridSize}, 50px)`;
        fillPolicyGrid(policyGridContainer, policy);

        row.appendChild(valueGridContainer);
        row.appendChild(policyGridContainer);
        viContainer.appendChild(row);

        // 建立另一個結果列，用於以動畫方式逐步標記最佳路徑
        const bestPathRow = document.createElement("div");
        bestPathRow.classList.add("result-row");

        const bestPathGrid = document.createElement("div");
        bestPathGrid.classList.add("grid");
        bestPathGrid.style.gridTemplateColumns = `repeat(${gridSize}, 50px)`;
        // 先填充政策網格，再以動畫標記最佳路徑
        fillPolicyGrid(bestPathGrid, policy);
        animateBestPath(bestPathGrid, policy);

        bestPathRow.appendChild(bestPathGrid);
        viContainer.appendChild(bestPathRow);
    })
    .catch(error => {
        console.error("Error computing policy:", error);
    });
}

function fillValueGrid(container, valueFunction) {
    for (let i = 0; i < gridSize * gridSize; i++) {
        const cell = document.createElement("div");
        cell.classList.add("cell");
        const row = Math.floor(i / gridSize);
        const col = i % gridSize;
        const key = `${row},${col}`;
        if (obstaclesList.includes(i)) {
            cell.textContent = "X";
            cell.style.backgroundColor = "black";
            cell.style.color = "white";
        } else if (i === startIndex) {
            cell.textContent = "S";
            cell.style.backgroundColor = "green";
            cell.style.color = "white";
        } else if (i === endIndex) {
            cell.textContent = "G";
            cell.style.backgroundColor = "red";
            cell.style.color = "white";
        } else if (valueFunction[key] !== undefined) {
            cell.textContent = parseFloat(valueFunction[key]).toFixed(2);
        } else {
            cell.textContent = "";
        }
        container.appendChild(cell);
    }
}

function fillPolicyGrid(container, policy) {
    for (let i = 0; i < gridSize * gridSize; i++) {
        const cell = document.createElement("div");
        cell.classList.add("cell");
        const row = Math.floor(i / gridSize);
        const col = i % gridSize;
        const key = `${row},${col}`;
        if (obstaclesList.includes(i)) {
            cell.textContent = "X";
            cell.style.backgroundColor = "black";
            cell.style.color = "white";
        } else if (i === startIndex) {
            cell.textContent = "S";
            cell.style.backgroundColor = "green";
            cell.style.color = "white";
        } else if (i === endIndex) {
            cell.textContent = "G";
            cell.style.backgroundColor = "red";
            cell.style.color = "white";
        } else if (policy[key]) {
            cell.textContent = policy[key];
        } else {
            cell.textContent = "";
        }
        container.appendChild(cell);
    }
}

function animateBestPath(gridElement, policy) {
    if (startIndex === null || endIndex === null) return;
    let current = startIndex;
    const visited = new Set();
    visited.add(current);
    function step() {
        markCellGreen(gridElement, current);
        if (current === endIndex) return;
        const row = Math.floor(current / gridSize);
        const col = current % gridSize;
        const key = `${row},${col}`;
        let next;
        switch (policy[key]) {
            case '↑':
                next = current - gridSize;
                break;
            case '↓':
                next = current + gridSize;
                break;
            case '←':
                next = current - 1;
                break;
            case '→':
                next = current + 1;
                break;
            case 'GOAL':
                next = endIndex;
                break;
            default:
                return;
        }
        if (visited.has(next)) return;
        visited.add(next);
        current = next;
        setTimeout(step, 500);
    }
    step();
}

function markCellGreen(gridElement, idx) {
    const cell = gridElement.children[idx];
    if (!cell) return;
    if (cell.textContent === "S" || cell.textContent === "G" || cell.textContent === "X") return;
    cell.style.backgroundColor = "lightgreen";
}
function generateRandomPolicy() {
    if (!gridSize) {
        alert("Please generate the grid first.");
        return;
    }
    const actions = ['↑', '↓', '←', '→'];
    const randomPolicy = {};
    for (let i = 0; i < gridSize * gridSize; i++) {
        const row = Math.floor(i / gridSize);
        const col = i % gridSize;
        const key = `${row},${col}`;
        if (obstaclesList.includes(i)) {
            randomPolicy[key] = "X";
        } else if (i === startIndex) {
            randomPolicy[key] = "S";
        } else if (i === endIndex) {
            randomPolicy[key] = "G";
        } else {
            randomPolicy[key] = actions[Math.floor(Math.random() * actions.length)];
        }
    }
    
    // 使用 generateRandomValueMatrix() 產生 value matrix (依你先前設定的隨機負值範圍)
    const valueMatrix = generateRandomValueMatrix();
    
    // 清空隨機結果容器 (只更新 #random-grid-container)
    const randomContainer = document.getElementById("random-grid-container");
    randomContainer.innerHTML = "";
    
    // 建立一個父容器，讓兩個欄位並排顯示
    const rowContainer = document.createElement("div");
    rowContainer.classList.add("random-row");
    
    // 建立「policy matrix」欄位容器
    const policyColumn = document.createElement("div");
    policyColumn.classList.add("random-column");
    const policyLabel = document.createElement("h3");
    policyLabel.textContent = "policy matrix";
    const policyGrid = document.createElement("div");
    policyGrid.classList.add("grid");
    policyGrid.style.gridTemplateColumns = `repeat(${gridSize}, 50px)`;
    // 填入 policyGrid 的格子
    for (let i = 0; i < gridSize * gridSize; i++) {
        const cell = document.createElement("div");
        cell.classList.add("cell");
        const row = Math.floor(i / gridSize);
        const col = i % gridSize;
        const key = `${row},${col}`;
        if (obstaclesList.includes(i)) {
            cell.textContent = "X";
            cell.style.backgroundColor = "black";
            cell.style.color = "white";
        } else if (i === startIndex) {
            cell.textContent = "S";
            cell.style.backgroundColor = "green";
            cell.style.color = "white";
        } else if (i === endIndex) {
            cell.textContent = "G";
            cell.style.backgroundColor = "red";
            cell.style.color = "white";
        } else {
            cell.textContent = randomPolicy[key];
        }
        policyGrid.appendChild(cell);
    }
    policyColumn.appendChild(policyLabel);
    policyColumn.appendChild(policyGrid);
    
    // 建立「value matrix」欄位容器
    const valueColumn = document.createElement("div");
    valueColumn.classList.add("random-column");
    const valueLabel = document.createElement("h3");
    valueLabel.textContent = "value matrix";
    const valueGrid = document.createElement("div");
    valueGrid.classList.add("grid");
    valueGrid.style.gridTemplateColumns = `repeat(${gridSize}, 50px)`;
    // 填入 valueGrid 的格子
    for (let i = 0; i < gridSize * gridSize; i++) {
        const cell = document.createElement("div");
        cell.classList.add("cell");
        const row = Math.floor(i / gridSize);
        const col = i % gridSize;
        const key = `${row},${col}`;
        if (obstaclesList.includes(i)) {
            cell.textContent = "X";
            cell.style.backgroundColor = "black";
            cell.style.color = "white";
        } else if (i === startIndex) {
            cell.textContent = "S";
            cell.style.backgroundColor = "green";
            cell.style.color = "white";
        } else if (i === endIndex) {
            cell.textContent = "G";
            cell.style.backgroundColor = "red";
            cell.style.color = "white";
        } else if (valueMatrix[key] !== undefined) {
            cell.textContent = valueMatrix[key];
        } else {
            cell.textContent = "";
        }
        valueGrid.appendChild(cell);
    }
    valueColumn.appendChild(valueLabel);
    valueColumn.appendChild(valueGrid);
    
    // 將兩個欄位加入父容器
    rowContainer.appendChild(policyColumn);
    rowContainer.appendChild(valueColumn);
    
    // 最後將父容器加入到隨機結果容器中
    randomContainer.appendChild(rowContainer);
}


function evaluateRandomPolicy(randomPolicy) {
    const gamma = 0.99;
    const tolerance = 0.001;
    let V = {};
    // 初始化所有非障礙物狀態的 V(s) = 0
    for (let i = 0; i < gridSize * gridSize; i++) {
        if (obstaclesList.includes(i)) continue;
        let row = Math.floor(i / gridSize);
        let col = i % gridSize;
        let key = `${row},${col}`;
        V[key] = 0;
    }
    let diff = Infinity;
    while (diff > tolerance) {
        diff = 0;
        let newV = { ...V };
        for (let i = 0; i < gridSize * gridSize; i++) {
            if (obstaclesList.includes(i)) continue;
            let row = Math.floor(i / gridSize);
            let col = i % gridSize;
            let key = `${row},${col}`;
            // 如果是終點，V(s)固定為 0
            if (i === endIndex) {
                newV[key] = 0;
                continue;
            }
            let action = randomPolicy[key];
            if (!action) continue;
            let next;
            switch (action) {
                case '↑':
                    next = i - gridSize;
                    break;
                case '↓':
                    next = i + gridSize;
                    break;
                case '←':
                    next = i - 1;
                    break;
                case '→':
                    next = i + 1;
                    break;
                case 'GOAL':
                    next = endIndex;
                    break;
                default:
                    next = i;
            }
            // 檢查 next 狀態是否合法
            let reward;
            if (next < 0 || next >= gridSize * gridSize || obstaclesList.includes(next)) {
                reward = -1; // 非法移動
                next = i; // 留在原地
            } else {
                reward = (next === endIndex) ? 1 : -0.04;
            }
            let nextRow = Math.floor(next / gridSize);
            let nextCol = next % gridSize;
            let nextKey = `${nextRow},${nextCol}`;
            newV[key] = reward + gamma * V[nextKey];
            diff = Math.max(diff, Math.abs(newV[key] - V[key]));
        }
        V = newV;
    }
    return V;
}
/**
 * 產生一個「負值到小正值」的 value matrix，
 * 預設範圍大約在 [-4, 1.24] 之間。
 */
function generateRandomValueMatrix() {
    const minVal = -4.0;     // 你可以視需要調整
    const maxVal = 1.24;     // 你也可改成 2.0 或其他上限
    let randomValueMatrix = {};

    for (let i = 0; i < gridSize * gridSize; i++) {
        if (obstaclesList.includes(i)) continue;  // 障礙物不給數值
        const row = Math.floor(i / gridSize);
        const col = i % gridSize;
        const key = `${row},${col}`;

        if (i === startIndex) {
            // 若想保留 "S" 不顯示數值，可不設定 randomValueMatrix[key]
            // randomValueMatrix[key] = 0; // or do nothing
        } else if (i === endIndex) {
            // 同理，若想保留 "G" 不顯示數值，可不設定
        } else {
            // 產生 [minVal, maxVal] 的亂數
            const val = Math.random() * (maxVal - minVal) + minVal;
            randomValueMatrix[key] = parseFloat(val.toFixed(2));
        }
    }
    return randomValueMatrix;
}
