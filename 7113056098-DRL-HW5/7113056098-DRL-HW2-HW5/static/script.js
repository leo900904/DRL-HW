let gridSize = 5;
let startSet = false;
let endSet = false;
let obstacles = 0;
let maxObstacles = 0;
let start_index = -1;
let end_index = -1;
let obstacle_indices = [];

// Helper function to create grid cells in a given container
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

        // Add specific classes for start, end, obstacles if applicable
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

    // Clear previous results
    document.getElementById("random-policy-grid").innerHTML = "";
    document.getElementById("random-value-grid").innerHTML = "";
    document.getElementById("optimal-policy-grid").innerHTML = "";
    document.getElementById("optimal-value-grid").innerHTML = "";
}

function handleCellClick(cell) {
    const index = parseInt(cell.dataset.index);
    if (!startSet) {
        // Clear previous start if any
        const currentStart = document.querySelector('.start');
        if (currentStart) currentStart.classList.remove('start');
        cell.classList.add("start");
        startSet = true;
        start_index = index;
    } else if (!endSet && !cell.classList.contains("start")) {
        // Clear previous end if any
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
        'terminal': 'X' // 終點或障礙物
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

    // For random policy, we can generate random values for visualization purposes
    const valueFunction = Array.from({length: gridSize * gridSize}, () => parseFloat((Math.random() * 5 - 2).toFixed(2))); // Random values between -2 and 3

    displayPolicyInContainer("random-policy-grid", policy, gridSize);
    displayValueFunctionInContainer("random-value-grid", valueFunction, gridSize);
}

function computePolicy() {
    if (start_index === -1 || end_index === -1) {
        alert("Please set start and end points first.");
        return;
    }

    fetch('/update_grid', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            gridSize: gridSize,
            startIndex: start_index,
            endIndex: end_index,
            obstacleIndices: obstacle_indices
        })
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.error || 'Failed to compute policy');
            });
        }
        return response.json();
    })
    .then(data => {
        if (data.message === "Grid updated successfully!") {
            const valueFunction = data.valueFunction;
            const policy = data.policy;
            displayValueFunctionInContainer("optimal-value-grid", valueFunction, gridSize);
            displayPolicyInContainer("optimal-policy-grid", policy, gridSize);
        } else {
            throw new Error(data.error || "Failed to compute policy");
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert(error.message || "An error occurred while computing the policy.");
    });
}

// Initial grid generation on page load
document.addEventListener('DOMContentLoaded', () => {
    generateGrid();
});
