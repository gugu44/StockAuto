import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from trading_env import StockTradingEnv
from utils import evaluate_performance


def load_prices(csv_path):
    df = pd.read_csv(csv_path)
    return df['Close'].values


def main(model_path, price_path):
    prices = load_prices(price_path)
    env = DummyVecEnv([lambda: StockTradingEnv(prices, window_size=30)])
    model = PPO.load(model_path)
    obs = env.reset()
    done = False
    rewards = []
    net_worths = []
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        rewards.append(reward[0])
        net_worths.append(env.envs[0].net_worth)
    evaluate_performance(net_worths)


if __name__ == '__main__':
    import sys
    main(sys.argv[1], sys.argv[2])

