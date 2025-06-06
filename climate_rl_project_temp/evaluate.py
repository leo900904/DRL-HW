"""
evaluate.py
──────────────────────────────────────────────────────
• 讀取 decision_log_PPO.csv & decision_log_SAC.csv
• 產生豐富視覺化：
  1) 政策強度熱圖
  2) 指標趨勢折線
  3) reward 收斂
  4) reward 變異度
  5) 行為變化率
  6) 投資-風險雙軸
  7) 平均政策雷達圖
  8) 全部圖彙總 overview
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import pi
from pathlib import Path
import matplotlib.image as mpimg

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False

# 檢查檔案
csv_paths = list(Path(".").glob("decision_log_*.csv"))
if not csv_paths:
    raise FileNotFoundError("找不到 decision_log_*.csv，請先執行 train_PPO.py 或 train_SAC.py")

for csv in csv_paths:
    algo = csv.stem.split("_")[-1]  # PPO or SAC
    df = pd.read_csv(csv)

    # 1️⃣ 政策強度熱圖
    actions = df[["Tax", "GreenSub", "CapStrict", "IndSub", "MixFund"]].values.T
    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(actions, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(range(5))
    ax.set_yticklabels(["碳稅", "綠補", "限排", "產補", "綜合"])
    ax.set_xlabel("Year")
    ax.set_title(f"{algo} 政策強度熱圖")
    cbar = plt.colorbar(im, ax=ax, fraction=0.025)
    cbar.set_label("Intensity (0~1)")
    plt.tight_layout()
    fig.savefig(f"heatmap_{algo}.png", dpi=300)
    plt.close(fig)
    print(f"[✓] heatmap_{algo}.png saved")

    # 2️⃣ 指標折線
    indicators = ["GDP", "Annual_CO2", "Temperature", "Support"]
    fig, ax = plt.subplots(figsize=(8, 4))
    for ind in indicators:
        ax.plot(df["Year"], df[ind], label=ind)
    ax.set_xlabel("Year")
    ax.set_title(f"{algo} 指標趨勢")
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"trend_{algo}.png", dpi=300)
    plt.close(fig)
    print(f"[✓] trend_{algo}.png saved")

    # 3️⃣ reward 收斂
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(df["Year"], df["Reward"])
    ax.set_xlabel("Year")
    ax.set_title(f"{algo} Reward 收斂")
    plt.tight_layout()
    fig.savefig(f"reward_{algo}.png", dpi=300)
    plt.close(fig)
    print(f"[✓] reward_{algo}.png saved")

    # 4️⃣ reward 變異度 (移動平均 ±1σ)
    df["Reward_MA"] = df["Reward"].rolling(window=5, min_periods=1).mean()
    df["Reward_STD"] = df["Reward"].rolling(window=5, min_periods=1).std()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(df["Year"], df["Reward_MA"], label="移動平均")
    ax.fill_between(df["Year"],
                     df["Reward_MA"] - df["Reward_STD"],
                     df["Reward_MA"] + df["Reward_STD"],
                     color="gray", alpha=0.3, label="±1σ")
    ax.set_xlabel("Year")
    ax.set_title(f"{algo} Reward 收斂穩定度")
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"reward_var_{algo}.png", dpi=300)
    plt.close(fig)
    print(f"[✓] reward_var_{algo}.png saved")

    # 5️⃣ 行為轉變率
    action_vectors = df[["Tax", "GreenSub", "CapStrict", "IndSub", "MixFund"]].values
    deltas = np.linalg.norm(np.diff(action_vectors, axis=0), axis=1)
    deltas = np.insert(deltas, 0, 0)  # 補上第一年 0
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(df["Year"], deltas)
    ax.set_xlabel("Year")
    ax.set_title(f"{algo} 年度政策變化幅度")
    plt.tight_layout()
    fig.savefig(f"action_delta_{algo}.png", dpi=300)
    plt.close(fig)
    print(f"[✓] action_delta_{algo}.png saved")

    # 6️⃣ 投資 - 風險雙軸
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(df["Year"], df["Investment"], color="green", label="投資")
    ax1.set_ylabel("Investment", color="green")

    ax2 = ax1.twinx()
    ax2.plot(df["Year"], df["Risk"], color="red", label="風險")
    ax2.set_ylabel("Risk", color="red")
    ax1.set_xlabel("Year")
    ax1.set_title(f"{algo} 投資-風險動態平衡")
    plt.tight_layout()
    fig.savefig(f"inv_risk_{algo}.png", dpi=300)
    plt.close(fig)
    print(f"[✓] inv_risk_{algo}.png saved")

    # 7️⃣ 平均政策雷達圖
    mean_actions = df[["Tax", "GreenSub", "CapStrict", "IndSub", "MixFund"]].mean()
    labels = ["碳稅", "綠補", "限排", "產補", "綜合"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    stats = mean_actions.tolist()
    stats += stats[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(subplot_kw={"polar": True}, figsize=(4, 4))
    ax.plot(angles, stats, linewidth=2)
    ax.fill(angles, stats, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_title(f"{algo} 平均政策強度雷達圖")
    plt.tight_layout()
    fig.savefig(f"radar_{algo}.png", dpi=300)
    plt.close(fig)
    print(f"[✓] radar_{algo}.png saved")

    # 8️⃣ 全部圖彙總 overview
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    img_files = [
        f"heatmap_{algo}.png",
        f"trend_{algo}.png",
        f"reward_{algo}.png",
        f"reward_var_{algo}.png",
        f"action_delta_{algo}.png",
        f"inv_risk_{algo}.png",
        f"radar_{algo}.png"
    ]
    for i, img_file in enumerate(img_files):
        img = mpimg.imread(img_file)
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(img_file.replace(f"_{algo}.png", ""))

    axes[-1].axis("off")
    fig.suptitle(f"{algo} 全部視覺化總覽", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    fig.savefig(f"overview_{algo}.png", dpi=300)
    plt.close(fig)
    print(f"[✓] overview_{algo}.png saved")

print("\n🎯 所有視覺化圖表與總覽圖已完成！報告可直接使用！")
