let gridSize = 5;
let startSet = false;
let endSet = false;
let obstacles = 0;
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
    obstacles = 0;

    const container = document.getElementById("grid-container");
    container.innerHTML = "";
    container.style.gridTemplateColumns = `repeat(${gridSize}, 50px)`;

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
    if (!startSet) {
        cell.classList.add("start");
        startSet = true;
    } else if (!endSet && !cell.classList.contains("start")) {
        cell.classList.add("end");
        endSet = true;
    } else if (obstacles < maxObstacles && !cell.classList.contains("start") && !cell.classList.contains("end")) {
        cell.classList.add("obstacle");
        obstacles++;
    }
}
