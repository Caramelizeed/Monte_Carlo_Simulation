"""
simulation.py
Geometric Brownian Motion simulation via exact Euler-Maruyama discretisation.
"""

import numpy as np


def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    dt: float,
    num_simulations: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate Geometric Brownian Motion paths using exact log-normal discretisation.

    The GBM SDE is:  dS = mu*S*dt + sigma*S*dW
    Exact solution:  S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)

    Parameters
    ----------
    S0              : float   — initial asset price
    mu              : float   — annualised drift (expected return)
    sigma           : float   — annualised volatility
    T               : float   — time horizon in years
    dt              : float   — time step in years (e.g. 1/252 for daily)
    num_simulations : int     — number of Monte Carlo paths
    seed            : int | None — random seed for reproducibility

    Returns
    -------
    paths : np.ndarray, shape (num_simulations, num_steps + 1)
        Each row is one simulated price path.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    num_steps = int(round(T / dt))
    drift     = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    # Draw all random increments at once (vectorised)
    Z = rng.standard_normal((num_simulations, num_steps))

    # Compute log-returns then cumsum for efficiency
    log_returns = drift + diffusion * Z                    # (N, steps)
    log_paths   = np.concatenate(
        [np.zeros((num_simulations, 1)), np.cumsum(log_returns, axis=1)],
        axis=1,
    )
    paths = S0 * np.exp(log_paths)                        # (N, steps+1)
    return paths


def final_prices(paths: np.ndarray) -> np.ndarray:
    """Return the terminal price of each simulated path."""
    return paths[:, -1]