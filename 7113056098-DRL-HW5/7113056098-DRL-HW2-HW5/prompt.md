# 專案重現說明

這個說明文件旨在幫助您在另一個聊天室中完整重現一個名為 "Grid World with Value Iteration & Best Path" 的 Flask 專案，僅提供詳細的文字說明。

## 1. 專案結構

專案的檔案結構如下：

```
c:/Users/allen/Desktop/DRL-HW/7113056098-DRL-HW1-2/
├── app.py
├── static/
│   ├── script.js
│   └── style.css
└── templates/
    └── index.html
```

## 2. 檔案功能說明

以下是每個檔案的功能說明：

### app.py

`app.py` 是 Flask 應用程式的主要檔案，負責處理路由、使用者請求和後端邏輯。

*   **路由 `/`**:  當使用者訪問網站根目錄時，會渲染 `index.html` 模板。
*   **路由 `/update_grid` (POST)**: 接收來自前端的 JSON 資料，用於更新網格狀態。
*   **`value_iteration_animate(grid_size, start_index, end_index, obstacles)` 函數**: 
    *   這個函數實現了價值迭代演算法，用於計算最佳策略和價值函數。
    *   它接收網格大小、起點、終點和障礙物作為輸入。
    *   它會迭代計算每個狀態的價值，直到收斂。
    *   它會記錄每次迭代的價值函數快照，用於動畫呈現。
    *   它會返回最佳策略、最終的價值函數和快照。
*   **路由 `/animate_policy` (POST)**: 接收來自前端的網格設定，並調用 `value_iteration_animate` 函數來計算策略和價值函數，然後將結果以 JSON 格式返回給前端，用於動畫顯示。
*   **`value_iteration(grid_size, start_index, end_index, obstacles)` 函數**:
    *   這個函數是 `value_iteration_animate` 的簡化版本，不包含動畫快照。
    *   它用於計算最佳策略和價值函數，但不提供動畫功能。
*   **路由 `/compute_policy` (POST)**: 接收來自前端的網格設定，並調用 `value_iteration` 函數來計算策略和價值函數，然後將結果以 JSON 格式返回給前端。
*   **`if __name__ == '__main__':`**:  啟動 Flask 應用程式，並啟用 debug 模式。

### static/script.js

`script.js` 檔案包含前端 JavaScript 程式碼，負責處理使用者互動、網格生成、價值迭代計算和動畫顯示。

*   **變數宣告**:  宣告了網格大小、起點、終點、障礙物等變數。
*   **`generateGrid()` 函數**:
    *   根據使用者輸入的網格大小，生成網格。
    *   清空現有的網格，並重新建立網格元素。
    *   為每個網格單元格添加點擊事件監聽器。
*   **`handleCellClick(cell)` 函數**:
    *   處理網格單元格的點擊事件。
    *   允許使用者設定起點、終點和障礙物。
*   **`computePolicy()` 函數**:
    *   獲取使用者設定的網格資訊。
    *   向後端發送 POST 請求，請求計算策略。
    *   接收後端返回的策略和價值函數。
    *   在網頁上顯示價值函數和策略網格。
    *   使用動畫顯示最佳路徑。
*   **`fillValueGrid(container, valueFunction)` 函數**:
    *   根據價值函數，填充價值網格。
    *   顯示每個單元格的價值。
    *   標記起點、終點和障礙物。
*   **`fillPolicyGrid(container, policy)` 函數**:
    *   根據策略，填充策略網格。
    *   顯示每個單元格的最佳動作。
    *   標記起點、終點和障礙物。
*   **`animateBestPath(gridElement, policy)` 函數**:
    *   使用動畫顯示最佳路徑。
    *   根據策略，逐步標記最佳路徑上的單元格。
*   **`markCellGreen(gridElement, idx)` 函數**:
    *   將指定單元格的背景顏色設為綠色。
*   **`generateRandomPolicy()` 函數**:
    *   生成隨機策略。
    *   生成隨機價值矩陣。
    *   在網頁上顯示隨機策略和價值矩陣。
*   **`evaluateRandomPolicy(randomPolicy)` 函數**:
    *   評估隨機策略。
    *   計算每個狀態的價值。
*   **`generateRandomValueMatrix()` 函數**:
    *   產生一個「負值到小正值」的 value matrix，
    *   預設範圍大約在 [-4, 1.24] 之間。

### static/style.css

`style.css` 檔案包含 CSS 樣式，用於設定網頁的外觀和佈局。

*   **`body`**: 設定字體、背景顏色、文字顏色和文字對齊方式。
*   **`#centered-container`**:  設定主要內容容器的樣式，包括外邊距、最大寬度、內邊距和對齊方式。
*   **.random-row, .result-row**: 設定結果列的樣式，包括 flex 佈局和間距。
*   **.random-column**: 設定結果欄位的樣式，包括 flex 佈局和對齊方式。
*   **`#button-row`**: 設定按鈕行的樣式，包括 flex 佈局和間距。
*   **`#input-grid-container`**: 設定輸入網格容器的樣式。
*   **`#grid-container`**: 設定網格容器的樣式，包括網格佈局、間距和外邊距。
*   **`#vi-results-container`**: 設定價值迭代結果容器的樣式。
*   **`.cell`**: 設定網格單元格的樣式，包括寬度、高度、邊框、字體大小、對齊方式、游標和過渡效果。
*   **.start, .end, .obstacle**: 設定起點、終點和障礙物的樣式，包括背景顏色和文字顏色。
*   **`button`**: 設定按鈕的樣式，包括內邊距、邊框、圓角、背景顏色、文字顏色、字體大小、游標和過渡效果。
*   **`button:hover`**: 設定按鈕懸停時的樣式，包括縮放和陰影效果。

### templates/index.html

`index.html` 檔案包含 HTML 結構，用於呈現網頁內容。

*   **`<!DOCTYPE html>`**:  聲明 HTML5 文件類型。
*   **`<html lang="en">`**:  HTML 根元素，指定語言為英文。
*   **`<head>`**:  包含網頁的元資料，包括字符集、視口設定和標題。
    *   **`<title>`**:  設定網頁標題。
    *   **`<link>`**:  連結外部 CSS 樣式表。
*   **`<body>`**:  包含網頁的可見內容。
    *   **`<div id="centered-container">`**:  主要內容容器，用於居中顯示內容。
        *   **`<h1>`**:  顯示網頁標題。
        *   **`<label>` 和 `<input>`**:  用於輸入網格大小。
        *   **`<div id="button-row">`**:  包含生成網格和計算策略的按鈕。
        *   **`<div id="input-grid-container">`**:  包含輸入網格。
            *   **`<h2>`**:  顯示輸入網格標題。
            *   **`<p>`**:  提供關於如何設定網格的說明。
            *   **`<div id="grid-container">`**:  顯示網格。
        *   **`<div id="random-results-container">`**: 包含隨機動作結果
            *   **`<h2>`**:  顯示隨機動作結果標題。
            *   **`<p>`**:  提供關於隨機動作結果的說明。
            *   **`<div id="random-grid-container">`**:  顯示隨機動作結果。
        *   **`<h2>`**:  顯示價值迭代結果標題。
        *   **`<p>`**:  提供關於價值迭代結果的說明。
        *   **`<div id="vi-results-container">`**:  顯示價值迭代結果。
    *   **`<script>`**:  連結外部 JavaScript 檔案。

