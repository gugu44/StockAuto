import gym
from gym import spaces
import numpy as np

class StockTradingEnv(gym.Env):
    """A simple stock trading environment."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, prices, window_size=10, initial_balance=10000, fee=0.001, slippage=0.001):
        super().__init__()
        self.prices = np.array(prices)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.fee = fee
        self.slippage = slippage
        self.action_space = spaces.Discrete(3)  # hold, buy, sell
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size + 2,),
            dtype=np.float32,
        )
        self.reset()

    def _get_obs(self):
        window_prices = self.prices[self.current_step - self.window_size : self.current_step]
        obs = np.concatenate(
            [window_prices, [self.shares_held], [self.balance]]
        ).astype(np.float32)
        return obs

    def reset(self):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.shares_held = 0
        return self._get_obs()

    def step(self, action):
        prev_price = self.prices[self.current_step - 1]
        price = self.prices[self.current_step]
        reward = 0.0
        done = False

        if action == 1:  # buy one share
            cost = price * (1 + self.slippage) * (1 + self.fee)
            if self.balance >= cost:
                self.balance -= cost
                self.shares_held += 1
        elif action == 2 and self.shares_held > 0:  # sell one share
            revenue = price * (1 - self.slippage) * (1 - self.fee)
            self.balance += revenue
            self.shares_held -= 1

        self.current_step += 1
        if self.current_step >= len(self.prices):
            done = True
        self.net_worth = self.balance + self.shares_held * price

        reward = (self.net_worth - self.initial_balance) / self.initial_balance

        obs = self._get_obs()
        info = {}
        return obs, reward, done, info

    def render(self, mode="human"):
        print(
            f"Step: {self.current_step} Balance: {self.balance:.2f} Shares: {self.shares_held} Net Worth: {self.net_worth:.2f}"
        )

