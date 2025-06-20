# StockAuto

Example project demonstrating a reinforcement learning pipeline for stock trading.

## Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Training

```
python src/train.py
```

## Backtesting

```
python src/backtest.py ppo_stock_trading data/test_prices.csv
```

## Real Trading

Integration with a real broker API is demonstrated in `src/real_trading.py`.

