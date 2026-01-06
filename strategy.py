
import numpy as np
from scipy.optimize import minimize
from finance_math import portfolio_variance, sharpe_ratio


def min_variance(mean_returns, cov_matrix):
   
    n = len(mean_returns)
    init_guess = np.ones(n) / n
    bounds = tuple((0, 1) for _ in range(n))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)

    result = minimize(
        lambda w: portfolio_variance(w, cov_matrix),
        init_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return result.x


def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate=0.03):
    n = len(mean_returns)
    init_guess = np.ones(n) / n
    bounds = tuple((0, 1) for _ in range(n))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)

    result = minimize(
        lambda w: -sharpe_ratio(w, mean_returns, cov_matrix, risk_free_rate),
        init_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return result.x




