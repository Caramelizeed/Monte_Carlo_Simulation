"""
analytics.py
Risk and return statistics for a Monte Carlo price simulation.
All metrics follow industry-standard quant definitions.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class SimulationStats:
    """Container for all computed risk/return metrics."""
    mean_simulated:   float
    mean_analytical:  float
    median_simulated: float
    median_analytical: float
    std:              float
    variance:         float
    var_95:           float   # 5th-pct price  (VaR expressed as price level)
    cvar_95:          float   # expected price in the tail (CVaR / ES)
    var_95_loss:      float   # dollar loss: S0 - var_95
    cvar_95_loss:     float   # dollar loss: S0 - cvar_95
    prob_gain:        float   # P(S_T > S0)
    sharpe_proxy:     float   # (mu - rf) / sigma


# ---------------------------------------------------------------------------
# Individual calculators
# ---------------------------------------------------------------------------

def calculate_mean(final_prices: np.ndarray) -> float:
    """Arithmetic mean of terminal prices (Monte Carlo estimator of E[S_T])."""
    return float(np.mean(final_prices))


def calculate_variance(final_prices: np.ndarray) -> float:
    """Sample variance of terminal prices (Bessel-corrected, ddof=1)."""
    return float(np.var(final_prices, ddof=1))


def calculate_std(final_prices: np.ndarray) -> float:
    """Sample standard deviation of terminal prices."""
    return float(np.std(final_prices, ddof=1))


def calculate_var(final_prices: np.ndarray, confidence: float = 0.95) -> float:
    """
    Value at Risk expressed as the terminal price at the (1-confidence) quantile.

    A loss of (S0 - VaR) occurs with probability (1 - confidence).

    Parameters
    ----------
    final_prices : np.ndarray
    confidence   : float — confidence level, default 0.95

    Returns
    -------
    float — the price level below which losses exceed VaR
    """
    return float(np.percentile(final_prices, (1 - confidence) * 100))


def calculate_cvar(final_prices: np.ndarray, confidence: float = 0.95) -> float:
    """
    Conditional Value at Risk (CVaR) / Expected Shortfall (ES).

    The average terminal price in the worst (1 - confidence) fraction of paths.
    CVaR is a coherent risk measure and strictly dominates VaR for risk management.

    Parameters
    ----------
    final_prices : np.ndarray
    confidence   : float — confidence level, default 0.95

    Returns
    -------
    float — expected price conditional on being in the tail
    """
    threshold = calculate_var(final_prices, confidence)
    tail_prices = final_prices[final_prices <= threshold]
    if len(tail_prices) == 0:
        return threshold
    return float(np.mean(tail_prices))


def calculate_prob_gain(final_prices: np.ndarray, S0: float) -> float:
    """Empirical probability that the terminal price exceeds the initial price."""
    return float(np.mean(final_prices > S0))


def calculate_sharpe_proxy(mu: float, sigma: float, rf: float = 0.05) -> float:
    """
    Sharpe ratio proxy: (mu - rf) / sigma.
    Uses the risk-free rate rf (default 5%).
    """
    return (mu - rf) / sigma


# ---------------------------------------------------------------------------
# Analytical GBM moments
# ---------------------------------------------------------------------------

def analytical_mean(S0: float, mu: float, T: float) -> float:
    """E[S_T] = S0 * exp(mu * T)  under risk-neutral / physical measure."""
    return S0 * np.exp(mu * T)


def analytical_median(S0: float, mu: float, sigma: float, T: float) -> float:
    """Median[S_T] = S0 * exp((mu - 0.5*sigma^2) * T)  (log-normal median)."""
    return S0 * np.exp((mu - 0.5 * sigma**2) * T)


def analytical_std(S0: float, mu: float, sigma: float, T: float) -> float:
    """Std[S_T] = S0 * exp(mu*T) * sqrt(exp(sigma^2*T) - 1)."""
    return (
        S0
        * np.exp(mu * T)
        * np.sqrt(np.exp(sigma**2 * T) - 1)
    )


# ---------------------------------------------------------------------------
# Aggregate summary
# ---------------------------------------------------------------------------

def compute_all_stats(
    final_prices: np.ndarray,
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    confidence: float = 0.95,
    rf: float = 0.05,
) -> SimulationStats:
    """
    Compute the full suite of risk/return metrics in one call.

    Returns a SimulationStats dataclass for convenient downstream use.
    """
    var95  = calculate_var(final_prices, confidence)
    cvar95 = calculate_cvar(final_prices, confidence)

    return SimulationStats(
        mean_simulated    = calculate_mean(final_prices),
        mean_analytical   = analytical_mean(S0, mu, T),
        median_simulated  = float(np.median(final_prices)),
        median_analytical = analytical_median(S0, mu, sigma, T),
        std               = calculate_std(final_prices),
        variance          = calculate_variance(final_prices),
        var_95            = var95,
        cvar_95           = cvar95,
        var_95_loss       = S0 - var95,
        cvar_95_loss      = S0 - cvar95,
        prob_gain         = calculate_prob_gain(final_prices, S0),
        sharpe_proxy      = calculate_sharpe_proxy(mu, sigma, rf),
    )


# ---------------------------------------------------------------------------
# Convergence analysis
# ---------------------------------------------------------------------------

def convergence_analysis(
    final_prices: np.ndarray,
    checkpoints: list[int] | None = None,
) -> dict[str, list]:
    """
    Evaluate how key statistics stabilise as simulation count grows.

    Parameters
    ----------
    final_prices : np.ndarray — full set of terminal prices
    checkpoints  : list[int]  — simulation counts at which to evaluate;
                                defaults to log-spaced values up to len(final_prices)

    Returns
    -------
    dict with keys 'n', 'mean', 'std', 'var_95', 'cvar_95'
    """
    N = len(final_prices)
    if checkpoints is None:
        checkpoints = list(
            dict.fromkeys(
                [int(x) for x in np.logspace(np.log10(100), np.log10(N), 20)]
                + [N]
            )
        )
    checkpoints = sorted(c for c in checkpoints if 10 <= c <= N)

    result: dict[str, list] = {k: [] for k in ["n", "mean", "std", "var_95", "cvar_95"]}
    for n in checkpoints:
        sub = final_prices[:n]
        result["n"].append(n)
        result["mean"].append(calculate_mean(sub))
        result["std"].append(calculate_std(sub))
        result["var_95"].append(calculate_var(sub))
        result["cvar_95"].append(calculate_cvar(sub))

    return result