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
    // 只更新隨機網格部分，不清空整個 random-results-container (以免蓋掉標題文字)
    const randomContainer = document.getElementById("random-grid-container");
    randomContainer.innerHTML = "";
    const randomPolicyGrid = document.createElement("div");
    randomPolicyGrid.classList.add("grid");
    randomPolicyGrid.style.gridTemplateColumns = `repeat(${gridSize}, 50px)`;
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
    randomContainer.appendChild(randomPolicyGrid);
}
