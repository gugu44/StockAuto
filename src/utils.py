import numpy as np


def compute_total_return(net_worths):
    return (net_worths[-1] - net_worths[0]) / net_worths[0]


def compute_mdd(net_worths):
    array = np.array(net_worths)
    cummax = np.maximum.accumulate(array)
    drawdown = (array - cummax) / cummax
    return drawdown.min()


def compute_sharpe(returns, risk_free_rate=0.0):
    returns = np.array(returns)
    excess = returns - risk_free_rate
    if returns.std() == 0:
        return 0.0
    return np.sqrt(252) * excess.mean() / returns.std()


def evaluate_performance(net_worths):
    returns = np.diff(net_worths) / net_worths[:-1]
    total_return = compute_total_return(net_worths)
    mdd = compute_mdd(net_worths)
    sharpe = compute_sharpe(returns)
    print(f"Total Return: {total_return:.2%}")
    print(f"Max Drawdown: {mdd:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")

