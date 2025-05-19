---

## layout: default

---

# DRL HW2 & HW5 作業說明檔案
# [HW5-HW2-ROO CODE 處理過程YT連結https://youtu.be/sa-c9rnVPUU](https://youtu.be/sa-c9rnVPUU)
# [結合WEBSIM YT連結https://youtu.be/-hKyWAA8xHA](https://youtu.be/-hKyWAA8xHA)
## 一、作業概述

本次專案整合 **HW2** 與 **HW5** 的要求，並結合 **Roo Code** 工具協助開發。整體系統採用 **Flask** 作為後端伺服器，前端以 **HTML/CSS/JavaScript** 實現可視化互動，達成以下功能：

### 1. HW2：價值迭代算法（Value Iteration）

* **目標**：實作價值迭代演算法，推導並顯示每個狀態的最佳策略與價值函數。
* **要求**：

  1. 使用螢幕網格（grid）表示環境狀態。
  2. 實現 Value Iteration，收斂後得到每個狀態下的最佳動作。
  3. 在格子上顯示對應的價值 $V(s)$，並可切換顯示隨機政策或最佳政策。

### 2. HW5：互動式前端與 Roo Code 協助優化

* **互動**：

  * 使用者可透過點擊設定：

    * 起點 (Start)
    * 終點 (Goal)
    * 障礙物 (Obstacles)
  * 提供按鈕切換視圖：

    * 隨機政策 (Random Policy)
    * 價值函數 (Value Function)
    * 最佳政策 (Optimal Policy)
* **Roo Code**：

  * 使用 Roo Code 協助撰寫後端演算法與前端互動程式碼。
  * 優化迭代收斂檢查與網格渲染效能。


## 五、影片示範

* 請參考附檔 `7113056098-HW5.mkv`，展示介面互動及演算法結果。

---
