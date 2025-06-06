"""
climate_env.py
──────────────────────────────────────────────────────
進階但簡化的氣候-經濟 RL 環境（連續動作版）
狀態 :  GDP, 年排放量, 公眾支持度, 綠投, 風險,
        大氣 CO2(ppm), 溫度 ΔT(°C)
動作 :  5 維向量 ∈ [0,1]
        [碳稅率, 綠能補助, 排放上限嚴格度,
         傳統產業補貼, 綜合政策基金]
Episode 長度 : 50 年
"""

import gym
from gym import spaces
import numpy as np


class ClimateEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, seed: int | None = 42) -> None:
        super().__init__()
        if seed is not None:
            np.random.seed(seed)

        # ──────────────────────────────────────────────
        #  動作空間：5 維連續向量，各維 0~1 代表政策強度
        # ──────────────────────────────────────────────
        self.action_space: spaces.Box = spaces.Box(
            low=np.zeros(5, dtype=np.float32),
            high=np.ones(5, dtype=np.float32),
            dtype=np.float32,
        )

        # ──────────────────────────────────────────────
        #  狀態空間範圍
        # ──────────────────────────────────────────────
        self.observation_space: spaces.Box = spaces.Box(
            low=np.array([50, 20, 0, 0, 0, 350, 0], dtype=np.float32),
            high=np.array([200, 200, 100, 100, 100, 600, 5], dtype=np.float32),
            dtype=np.float32,
        )

        # 對應中文名稱，方便視覺化
        self.action_names = [
            "碳稅",
            "綠能補助",
            "限制排放",
            "產業補貼",
            "綜合基金",
        ]

        self.reset()

    # ──────────────────────────────
    #  reset：初始化環境
    # ──────────────────────────────
    def reset(self):
        self.gdp = 100.0
        self.annual_co2 = 100.0
        self.support = 50.0
        self.invest = 20.0
        self.risk = 10.0
        self.atm_co2 = 415.0
        self.temp = 1.0
        self.year = 0
        return self._get_state()

    # ──────────────────────────────
    #  step：推進一年
    # ──────────────────────────────
    def step(self, action: np.ndarray):
        # 取出 5 維強度
        tax, green_sub, cap_strict, ind_sub, mix_fund = action

        # 1️⃣ 政策直接效果（線性比例）
        self.annual_co2 += (
            -6.0 * tax        # 碳稅抑制排放
            -4.0 * green_sub  # 綠補
            -8.0 * cap_strict # 排放上限嚴格
            + 3.0 * ind_sub   # 補貼傳統產業 → 排放升
            -4.0 * mix_fund   # 綜合政策
        )
        self.gdp += (
            -3.0 * tax
            -1.0 * green_sub
            -4.0 * cap_strict
            + 3.0 * ind_sub
            -2.0 * mix_fund
        )
        self.support += (
            -4.0 * tax
            + 2.0 * green_sub
            -2.0 * cap_strict
            + 2.0 * ind_sub
            + 1.0 * mix_fund
        )
        self.invest += 5.0 * green_sub + 1.0 * mix_fund

        # 2️⃣ 綠投帶來額外減排（比例 3%）
        self.annual_co2 -= 0.03 * self.invest

        # 3️⃣ 隨機外部雜訊
        self.gdp += np.random.uniform(-0.4, 0.6)
        self.support += np.random.uniform(-1.0, 1.0)

        # 4️⃣ 大氣 CO₂ 更新：排放轉 ppm；自然吸收 0.5%
        delta_ppm = (self.annual_co2 - 100.0) * 0.05
        natural_sink = 0.005 * max(0.0, self.atm_co2 - 415.0)
        self.atm_co2 += delta_ppm - natural_sink

        # 5️⃣ 溫度：對數模式（Arrhenius 極簡版）
        self.temp = 1.0 + 0.01 * np.log(self.atm_co2 / 415.0)

        # 6️⃣ 溫度損害對 GDP（二次損失函數）
        if self.temp > 1.5:
            self.gdp -= 0.25 * (self.temp - 1.5) ** 2

        # 7️⃣ 風險：溫度 + CO₂ 線性加權
        self.risk = np.clip(self.temp * 20 + (self.atm_co2 - 415) * 0.1, 0, 100)

        # 8️⃣ 變數裁切
        self.annual_co2 = np.clip(self.annual_co2, 20, 200)
        self.gdp = np.clip(self.gdp, 50, 200)
        self.support = np.clip(self.support, 0, 100)
        self.invest = np.clip(self.invest, 0, 100)
        self.atm_co2 = np.clip(self.atm_co2, 350, 600)
        self.temp = np.clip(self.temp, 0, 5)

        # 9️⃣ reward（多目標線性 + 溫度二次懲罰）
        temp_penalty = 3.0 * max(0, self.temp - 1.5) ** 2
        reward = (
            +1.0 * (self.gdp - 100)
            - 1.0 * (self.annual_co2 - 100)
            + 0.5 * (self.support - 50)
            - 0.2 * self.risk
            - temp_penalty
        )

        # 10️⃣ 進入下一年
        self.year += 1
        done = self.year >= 50

        info = {"action_vector": action}
        return self._get_state(), reward, done, info

    # ------------------------------------------------------
    #  內部工具
    # ------------------------------------------------------
    def _get_state(self) -> np.ndarray:
        return np.array(
            [
                self.gdp,
                self.annual_co2,
                self.support,
                self.invest,
                self.risk,
                self.atm_co2,
                self.temp,
            ],
            dtype=np.float32,
        )

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
