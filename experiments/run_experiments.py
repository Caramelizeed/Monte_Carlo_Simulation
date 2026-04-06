"""
run_experiment.py
Entry point for the GBM Monte Carlo simulation experiment.

Usage:
    python run_experiment.py
    python run_experiment.py --S0 150 --mu 0.12 --sigma 0.30 --T 2 --N 10000
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from simulation   import simulate_gbm, final_prices
from analytics    import compute_all_stats, convergence_analysis, analytical_mean
from visualization import (
    plot_price_paths,
    plot_distribution,
    plot_convergence,
    plot_summary_dashboard,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GBM Monte Carlo simulation — quant risk analysis"
    )
    p.add_argument("--S0",    type=float, default=100.0,  help="Initial price")
    p.add_argument("--mu",    type=float, default=0.10,   help="Annualised drift")
    p.add_argument("--sigma", type=float, default=0.25,   help="Annualised volatility")
    p.add_argument("--T",     type=float, default=1.0,    help="Horizon in years")
    p.add_argument("--dt",    type=float, default=1/252,  help="Time step in years (default: 1 trading day)")
    p.add_argument("--N",     type=int,   default=10_000, help="Number of simulations")
    p.add_argument("--seed",  type=int,   default=42,     help="Random seed")
    p.add_argument("--conf",  type=float, default=0.95,   help="VaR / CVaR confidence level")
    p.add_argument("--rf",    type=float, default=0.05,   help="Risk-free rate for Sharpe proxy")
    p.add_argument("--outdir",type=str,   default="output",help="Directory for saved plots")
    p.add_argument("--no-plots", action="store_true",     help="Skip all plotting")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(stats, args) -> None:
    SEP = "─" * 54
    print(f"\n{'GBM MONTE CARLO — SIMULATION REPORT':^54}")
    print(SEP)
    print(f"  {'Parameter':<28} {'Value':>12}")
    print(SEP)
    params = [
        ("S₀ — initial price",    f"${args.S0:.2f}"),
        ("μ  — drift (ann.)",     f"{args.mu:.4f}"),
        ("σ  — volatility (ann.)",f"{args.sigma:.4f}"),
        ("T  — horizon (years)",  f"{args.T:.2f}"),
        ("dt — time step (years)",f"{args.dt:.6f}"),
        ("N  — simulations",      f"{args.N:,}"),
        ("Confidence level",      f"{args.conf:.0%}"),
        ("Risk-free rate",        f"{args.rf:.2%}"),
    ]
    for label, val in params:
        print(f"  {label:<28} {val:>12}")

    print(f"\n{'RISK & RETURN METRICS':^54}")
    print(SEP)
    metrics = [
        ("E[S_T] analytical",          f"${stats.mean_analytical:.4f}"),
        ("E[S_T] simulated",           f"${stats.mean_simulated:.4f}"),
        ("Median analytical",          f"${stats.median_analytical:.4f}"),
        ("Median simulated",           f"${stats.median_simulated:.4f}"),
        ("Std deviation σ[S_T]",       f"${stats.std:.4f}"),
        ("Variance Var[S_T]",          f"${stats.variance:.4f}"),
        (f"VaR {args.conf:.0%} (price level)",  f"${stats.var_95:.4f}"),
        (f"VaR {args.conf:.0%} (dollar loss)",  f"-${stats.var_95_loss:.4f}"),
        (f"CVaR {args.conf:.0%} (price level)", f"${stats.cvar_95:.4f}"),
        (f"CVaR {args.conf:.0%} (dollar loss)", f"-${stats.cvar_95_loss:.4f}"),
        ("P(S_T > S₀)",               f"{stats.prob_gain:.2%}"),
        ("Sharpe proxy",               f"{stats.sharpe_proxy:.4f}"),
    ]
    for label, val in metrics:
        print(f"  {label:<32} {val:>10}")

    print(SEP)
    bias = (stats.mean_simulated - stats.mean_analytical) / stats.mean_analytical
    print(f"  {'MC bias (simulated vs analytical)':<32} {bias:>+9.4%}")
    print(SEP + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print(f"\nRunning GBM Monte Carlo: N={args.N:,}  T={args.T}y  σ={args.sigma}")
    t0 = time.perf_counter()

    # 1. Simulate
    paths  = simulate_gbm(
        S0=args.S0, mu=args.mu, sigma=args.sigma,
        T=args.T, dt=args.dt, num_simulations=args.N, seed=args.seed,
    )
    finals = final_prices(paths)
    elapsed = time.perf_counter() - t0
    print(f"Simulation complete in {elapsed:.2f}s  |  paths shape: {paths.shape}")

    # 2. Compute statistics
    stats = compute_all_stats(
        final_prices=finals, S0=args.S0, mu=args.mu, sigma=args.sigma,
        T=args.T, confidence=args.conf, rf=args.rf,
    )
    conv = convergence_analysis(finals)
    ana_mean = analytical_mean(args.S0, args.mu, args.T)

    # 3. Print report
    print_report(stats, args)

    # 4. Plots
    if not args.no_plots:
        os.makedirs(args.outdir, exist_ok=True)

        plot_price_paths(
            paths, S0=args.S0,
            save_path=os.path.join(args.outdir, "price_paths.png"), show=False,
        )
        plot_distribution(
            finals, S0=args.S0, var_95=stats.var_95, cvar_95=stats.cvar_95,
            save_path=os.path.join(args.outdir, "distribution.png"), show=False,
        )
        plot_convergence(
            conv, analytical_mean=ana_mean,
            save_path=os.path.join(args.outdir, "convergence.png"), show=False,
        )
        plot_summary_dashboard(
            paths=paths, final_prices=finals, conv=conv,
            S0=args.S0, var_95=stats.var_95, cvar_95=stats.cvar_95,
            analytical_mean=ana_mean,
            save_path=os.path.join(args.outdir, "dashboard.png"), show=False,
        )
        print(f"Plots saved to ./{args.outdir}/")


if __name__ == "__main__":
    main()