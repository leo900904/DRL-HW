let gridSize = 5;
let startSet = false;
let endSet = false;
let startIndex = null;
let endIndex = null;
let obstaclesList = [];
let maxObstacles = 0;

function generateGrid() {
    gridSize = parseInt(document.getElementById("gridSize").value);
    if (isNaN(gridSize) || gridSize < 5 || gridSize > 9) {
        alert("Please enter a number between 5 and 9.");
        return;
    }

    maxObstacles = gridSize - 2;
    startSet = false;
    endSet = false;
    startIndex = null;
    endIndex = null;
    obstaclesList = [];

    // 清空輸入網格
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
    // 設定起點
    if (!startSet) {
        cell.classList.add("start");
        startSet = true;
        startIndex = index;
    }
    // 設定終點
    else if (!endSet && !cell.classList.contains("start")) {
        cell.classList.add("end");
        endSet = true;
        endIndex = index;
    }
    // 設定障礙物
    else if (obstaclesList.length < maxObstacles &&
             !cell.classList.contains("start") &&
             !cell.classList.contains("end")) {
        cell.classList.add("obstacle");
        obstaclesList.push(index);
    }
}

// 呼叫後端 Value Iteration 並更新 vi-results-container
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
        
        // 先清空 Value Iteration 結果的容器
        const viContainer = document.getElementById("vi-results-container");
        viContainer.innerHTML = "";

        // 建立一個結果列，橫向排列 Value Function 與 Policy 網格
        const row = document.createElement("div");
        row.classList.add("result-row");

        // 建立 Value Function 網格
        const valueGridContainer = document.createElement("div");
        valueGridContainer.classList.add("grid");
        valueGridContainer.style.gridTemplateColumns = `repeat(${gridSize}, 50px)`;
        fillValueGrid(valueGridContainer, valueFunction);

        // 建立 Policy 網格
        const policyGridContainer = document.createElement("div");
        policyGridContainer.classList.add("grid");
        policyGridContainer.style.gridTemplateColumns = `repeat(${gridSize}, 50px)`;
        fillPolicyGrid(policyGridContainer, policy);

        // 加入兩個網格到同一橫列中
        row.appendChild(valueGridContainer);
        row.appendChild(policyGridContainer);
        viContainer.appendChild(row);

        // 建立另一個結果列顯示最佳路徑
        const bestPathRow = document.createElement("div");
        bestPathRow.classList.add("result-row");

        const bestPathGrid = document.createElement("div");
        bestPathGrid.classList.add("grid");
        bestPathGrid.style.gridTemplateColumns = `repeat(${gridSize}, 50px)`;
        // 先以 policy 填充網格，再標示最佳路徑
        fillPolicyGrid(bestPathGrid, policy);
        highlightBestPath(bestPathGrid, policy);

        bestPathRow.appendChild(bestPathGrid);
        viContainer.appendChild(bestPathRow);
    })
    .catch(error => {
        console.error("Error computing policy:", error);
    });
}

// 依據 valueFunction 填充網格 (用於 Value Iteration 結果)
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

// 依據 policy 填充網格 (用於 Policy 結果)
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

// 根據 policy 從 startIndex 依序走到 endIndex，並將沿途路徑標記成綠色 (最佳路徑)
function highlightBestPath(gridElement, policy) {
    if (startIndex === null || endIndex === null) return;

    let current = startIndex;
    const visited = new Set();
    visited.add(current);

    while (current !== endIndex) {
        const row = Math.floor(current / gridSize);
        const col = current % gridSize;
        const key = `${row},${col}`;
        const action = policy[key];

        // 將當前 cell 標記成綠色 (前提是不覆蓋 S/G/X)
        markCellGreen(gridElement, current);

        // 根據 action 判斷下一個 cell 的 index
        let next;
        switch (action) {
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
        if (visited.has(next)) {
            // 如果進入循環，停止標記
            return;
        }
        visited.add(next);
        current = next;
    }
}

// 將 gridElement 中指定 index 的 cell 標記成綠色 (但不覆蓋 S/G/X)
function markCellGreen(gridElement, idx) {
    const cell = gridElement.children[idx];
    if (!cell) return;
    if (cell.textContent === "S" || cell.textContent === "G" || cell.textContent === "X") return;
    cell.style.backgroundColor = "lightgreen";
}

// 產生隨機策略並更新 random-results-container
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

    // 先清空隨機結果容器
    const randomContainer = document.getElementById("random-results-container");
    randomContainer.innerHTML = "";

    // 建立一個結果列，用於隨機策略網格
    const row = document.createElement("div");
    row.classList.add("result-row");

    const randomPolicyGrid = document.createElement("div");
    randomPolicyGrid.classList.add("grid");
    randomPolicyGrid.style.gridTemplateColumns = `repeat(${gridSize}, 50px)`;

    // 填充隨機策略網格
    for (let i = 0; i < gridSize * gridSize; i++) {
        const cell = document.createElement("div");
        cell.classList.add("cell");
        const key = `${Math.floor(i / gridSize)},${i % gridSize}`;
        
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
        randomPolicyGrid.appendChild(cell);
    }
    row.appendChild(randomPolicyGrid);
    randomContainer.appendChild(row);
}
