import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from strategy import min_variance, max_sharpe_ratio


def fetch_data(tickers, period="3y"):
    """
    Download adjusted close prices and compute daily returns, mean returns, covariance.
    Requires yfinance installed.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required: pip install yfinance")
    data = yf.download(tickers, period=period)["Close"].dropna()
    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    return data, returns, mean_returns, cov_matrix


def plot_frontier(mean_returns, cov_matrix, strategy_name, risk_free_rate=0.03):
    n = len(mean_returns)
    bounds = tuple((0, 1) for _ in range(n))
    init_guess = n * [1. / n]


    # --- Min Variance Point ---
    cons_mv = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    res_mv = minimize(lambda w: np.dot(w.T, np.dot(cov_matrix, w)), init_guess,
                      bounds=bounds, constraints=cons_mv)
    w_mv = res_mv.x
    r_mv = np.dot(w_mv, mean_returns) * 252
    vol_mv = np.sqrt(np.dot(w_mv.T, np.dot(cov_matrix, w_mv))) * np.sqrt(252)

    w_sr = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    r_sr = np.dot(w_sr, mean_returns) * 252
    vol_sr = np.sqrt(np.dot(w_sr.T, np.dot(cov_matrix, w_sr))) * np.sqrt(252)

    # --- Build Frontier Curve ---
    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), 30)
    frontier_vols = []
    frontier_returns = []

    for r in target_returns:
        cons = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.dot(x, mean_returns) - r / 252}
        )
        res = minimize(lambda w: np.dot(w.T, np.dot(cov_matrix, w)), init_guess,
                       bounds=bounds, constraints=cons)
        if res.success:
            vol = np.sqrt(res.fun) * np.sqrt(252)
            frontier_vols.append(vol)
            frontier_returns.append(r)
        else:
            frontier_vols.append(np.nan)
            frontier_returns.append(r)

    # Add both min var and max Sharpe points to the curve
    frontier_vols += [vol_mv, vol_sr]
    frontier_returns += [r_mv, r_sr]

    # Sort by volatility for smooth line
    points = sorted(zip(frontier_vols, frontier_returns))
    vols_sorted, rets_sorted = zip(*points)

    # Optimal Point (red star)
    if strategy_name == "min_variance":
        opt_weights = w_mv
    else:
        opt_weights = w_sr
    opt_ret = np.dot(opt_weights, mean_returns) * 252
    opt_vol = np.sqrt(np.dot(opt_weights.T, np.dot(cov_matrix, opt_weights))) * np.sqrt(252)

    # --- Plot ---
    fig, ax = plt.subplots()
    ax.plot(vols_sorted, rets_sorted, label="Efficient Frontier")
    ax.scatter(opt_vol, opt_ret, marker="*", color="red", s=150, label="Optimal Portfolio")
    ax.set_xlabel("Volatility (Annualized)")
    ax.set_ylabel("Return (Annualized)")
    ax.legend()
    ax.grid(True)
    return fig
