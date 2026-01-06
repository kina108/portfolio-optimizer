from utils import fetch_data, plot_frontier
from finance_math import calc_stats


def optimize_portfolio(tickers, strategy_name, risk_free_rate=0.03):
    data, returns, mean_returns, cov_matrix = fetch_data(tickers)
    if strategy_name == "min_variance":
        from strategy import min_variance
        weights = min_variance(mean_returns, cov_matrix)
    else:
        from strategy import max_sharpe_ratio
        weights = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    stats = calc_stats(weights, mean_returns, cov_matrix, risk_free_rate)
    return weights, stats


def plot_efficient_frontier(tickers, strategy_name, risk_free_rate=0.03):
    data, returns, mean_returns, cov_matrix = fetch_data(tickers)
    return plot_frontier(mean_returns, cov_matrix, strategy_name, risk_free_rate)






