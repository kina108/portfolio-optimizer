import numpy as np


def expected_return(weights, mean_returns):
    return np.dot(weights, mean_returns) * 252


def portfolio_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights)) * 252


def sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    port_return = expected_return(weights, mean_returns)
    port_volatility = np.sqrt(portfolio_variance(weights, cov_matrix))
    return (port_return - risk_free_rate) / port_volatility


def calc_stats(weights, mean_returns, cov_matrix, risk_free_rate=0.03):
    port_return = expected_return(weights, mean_returns)
    port_vol = np.sqrt(portfolio_variance(weights, cov_matrix))
    sr = sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate)
    return {
        "Expected Return (Annualized)": f"{port_return:.2%}",
        "Volatility (Annualized)": f"{port_vol:.2%}",
        "Sharpe Ratio": f"{sr:.2f}",

    }
