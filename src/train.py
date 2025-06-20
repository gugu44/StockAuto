import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from trading_env import StockTradingEnv

# Example data loader
def load_prices(csv_path):
    df = pd.read_csv(csv_path)
    return df['Close'].values


def main():
    prices = load_prices('data/train_prices.csv')
    env = DummyVecEnv([lambda: StockTradingEnv(prices, window_size=30)])
    model = PPO('MlpPolicy', env, verbose=1, batch_size=32)
    model.learn(total_timesteps=10000)
    model.save('ppo_stock_trading')


if __name__ == '__main__':
    main()

