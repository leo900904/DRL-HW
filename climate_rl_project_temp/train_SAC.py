"""
train_SAC.py
──────────────────────────────────────────────────────
• 連續動作版 ClimateEnv
• 單獨訓練 SAC，200_000 timestep
• 評估 1 條 episode，紀錄動作向量與狀態
• 正確顯示進度條（off-policy 演算法適配）
"""

import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from climate_env import ClimateEnv
from tqdm import tqdm


class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress")

    def _on_step(self) -> bool:
        # SAC (off-policy) 沒有 n_steps，用 num_timesteps 直接更新
        self.pbar.update(1)
        return True

    def _on_training_end(self) -> None:
        self.pbar.close()


def run_and_log(model, env, algo: str):
    obs = env.reset()
    done = False
    log = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        log.append(
            {
                "Year": env.year,
                "Algo": algo,
                "Reward": reward,
                "GDP": env.gdp,
                "Annual_CO2": env.annual_co2,
                "Support": env.support,
                "Investment": env.invest,
                "Risk": env.risk,
                "Atm_CO2": env.atm_co2,
                "Temperature": env.temp,
                # 動作向量五維
                "Tax": info["action_vector"][0],
                "GreenSub": info["action_vector"][1],
                "CapStrict": info["action_vector"][2],
                "IndSub": info["action_vector"][3],
                "MixFund": info["action_vector"][4],
            }
        )
    pd.DataFrame(log).to_csv(
        f"decision_log_{algo}.csv", index=False, encoding="utf-8-sig"
    )
    print(f"[✓] decision_log_{algo}.csv saved")


if __name__ == "__main__":
    total_timesteps = 200_000

    env = ClimateEnv()
    model_sac = SAC("MlpPolicy", env, verbose=0, seed=42)
    print("Training SAC ...")
    model_sac.learn(
        total_timesteps=total_timesteps,
        callback=ProgressBarCallback(total_timesteps)
    )
    model_sac.save("sac_continuous")
    run_and_log(model_sac, env, "SAC")
