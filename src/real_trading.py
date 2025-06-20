"""Example real trading integration. Requires broker API libraries."""
import time
from trading_env import StockTradingEnv
from utils import evaluate_performance

# Placeholder for broker API
class BrokerAPI:
    def buy(self, ticker, quantity):
        print(f"Buying {quantity} shares of {ticker}")

    def sell(self, ticker, quantity):
        print(f"Selling {quantity} shares of {ticker}")

    def get_price(self, ticker):
        # Replace with real-time price
        return 100.0


def trade_loop(model, ticker, api, max_position=10, stop_loss=0.1, take_profit=0.2):
    prices = []
    net_worths = [10000]
    shares = 0
    balance = 10000
    step = 0
    while True:
        price = api.get_price(ticker)
        prices.append(price)
        if len(prices) < model.env.envs[0].window_size:
            time.sleep(1)
            continue
        env = StockTradingEnv(prices[-model.env.envs[0].window_size:], initial_balance=balance)
        obs = env._get_obs()
        action, _ = model.predict(obs, deterministic=True)
        if action == 1 and shares < max_position:
            api.buy(ticker, 1)
            shares += 1
            balance -= price
        elif action == 2 and shares > 0:
            api.sell(ticker, 1)
            shares -= 1
            balance += price

        net = balance + shares * price
        net_worths.append(net)
        if (net - net_worths[0]) / net_worths[0] <= -stop_loss:
            print("Stop loss triggered")
            break
        if (net - net_worths[0]) / net_worths[0] >= take_profit:
            print("Take profit triggered")
            break
        step += 1
        time.sleep(1)

    evaluate_performance(net_worths)


