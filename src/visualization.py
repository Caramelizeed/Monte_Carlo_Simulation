"""
visualization.py
Publication-quality plotting for GBM Monte Carlo simulation results.
"""

from __future__ import annotations
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

BLUE   = "#378ADD"
TEAL   = "#1D9E75"
CORAL  = "#D85A30"
AMBER  = "#BA7517"
RED    = "#E24B4A"
GRAY   = "#888780"
LIGHT  = "#F1EFE8"

def _apply_base_style(ax: plt.Axes, title: str = "") -> None:
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#D3D1C7")
    ax.tick_params(colors=GRAY, labelsize=9)
    ax.set_title(title, fontsize=10, fontweight="normal", color="#444441", pad=8)
    ax.grid(True, color=LIGHT, linewidth=0.8, zorder=0)


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

def plot_price_paths(
    paths: np.ndarray,
    S0: float | None = None,
    max_paths: int = 30,
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot a sample of simulated GBM price paths.

    Parameters
    ----------
    paths     : np.ndarray, shape (N, steps+1)
    S0        : float — initial price for a reference line
    max_paths : int   — maximum number of paths to display
    save_path : str   — optional file path to save PNG/PDF
    show      : bool  — call plt.show() after rendering
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_base_style(ax, "Simulated price paths (GBM)")

    n_plot = min(max_paths, paths.shape[0])
    for i in range(n_plot):
        alpha = 0.65 if i == 0 else 0.25
        lw    = 1.4  if i == 0 else 0.7
        color = BLUE if i == 0 else "#8AB8E6"
        ax.plot(paths[i], color=color, lw=lw, alpha=alpha, zorder=2)

    if S0 is not None:
        ax.axhline(S0, color=AMBER, lw=1.2, ls="--", label=f"S₀ = {S0:.2f}", zorder=3)
        ax.legend(fontsize=9, framealpha=0.6)

    ax.set_xlabel("Time step", fontsize=9, color=GRAY)
    ax.set_ylabel("Price ($)", fontsize=9, color=GRAY)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_distribution(
    final_prices: np.ndarray,
    S0: float | None = None,
    var_95: float | None = None,
    cvar_95: float | None = None,
    bins: int = 60,
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot histogram of terminal prices with optional VaR / CVaR annotations.

    Parameters
    ----------
    final_prices : np.ndarray
    S0           : float — initial price reference line
    var_95       : float — 95% VaR price level
    cvar_95      : float — 95% CVaR price level
    bins         : int   — histogram bin count
    save_path    : str
    show         : bool
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_base_style(ax, "Terminal price distribution")

    counts, edges, patches = ax.hist(
        final_prices, bins=bins, color=BLUE, alpha=0.75,
        edgecolor="white", linewidth=0.3, zorder=2,
    )

    # Colour the tail red
    if var_95 is not None:
        for patch, left in zip(patches, edges[:-1]):
            if left < var_95:
                patch.set_facecolor(RED)
                patch.set_alpha(0.8)

    legend_handles = []

    if S0 is not None:
        ax.axvline(S0, color=AMBER, lw=1.5, ls="--", zorder=4)
        legend_handles.append(Line2D([0], [0], color=AMBER, lw=1.5, ls="--", label=f"S₀ = {S0:.2f}"))

    if var_95 is not None:
        ax.axvline(var_95, color=RED, lw=1.5, ls="-", zorder=4)
        legend_handles.append(Line2D([0], [0], color=RED, lw=1.5, label=f"VaR 95% = {var_95:.2f}"))

    if cvar_95 is not None:
        ax.axvline(cvar_95, color=CORAL, lw=1.5, ls=":", zorder=4)
        legend_handles.append(Line2D([0], [0], color=CORAL, lw=1.5, ls=":", label=f"CVaR 95% = {cvar_95:.2f}"))

    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=9, framealpha=0.6)

    ax.set_xlabel("Terminal price ($)", fontsize=9, color=GRAY)
    ax.set_ylabel("Frequency", fontsize=9, color=GRAY)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_convergence(
    conv: dict[str, list],
    analytical_mean: float | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot convergence of mean, std, VaR and CVaR as simulation count grows.

    Parameters
    ----------
    conv             : dict from analytics.convergence_analysis()
    analytical_mean  : float — theoretical E[S_T] to overlay on the mean panel
    save_path        : str
    show             : bool
    """
    ns     = conv["n"]
    series = [
        ("mean",   "E[S_T] convergence",   BLUE,  analytical_mean),
        ("std",    "σ[S_T] convergence",    TEAL,  None),
        ("var_95", "VaR 95% convergence",   RED,   None),
        ("cvar_95","CVaR 95% convergence",  CORAL, None),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (key, title, color, ref) in zip(axes, series):
        _apply_base_style(ax, title)
        ax.plot(ns, conv[key], color=color, lw=1.6, marker="o", ms=3, zorder=3)
        if ref is not None:
            ax.axhline(ref, color=AMBER, lw=1.2, ls="--",
                       label=f"analytical = {ref:.2f}", zorder=2)
            ax.legend(fontsize=8, framealpha=0.6)
        ax.set_xlabel("Simulations (N)", fontsize=9, color=GRAY)
        ax.set_ylabel("$", fontsize=9, color=GRAY)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1f"))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.suptitle("Monte Carlo convergence diagnostics", fontsize=11,
                 fontweight="normal", color="#444441", y=1.01)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_summary_dashboard(
    paths: np.ndarray,
    final_prices: np.ndarray,
    conv: dict[str, list],
    S0: float,
    var_95: float,
    cvar_95: float,
    analytical_mean: float,
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Combined 2x2 dashboard: paths, distribution, mean convergence, VaR convergence.
    """
    fig = plt.figure(figsize=(14, 9))
    gs  = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.3)

    ax_paths = fig.add_subplot(gs[0, 0])
    ax_dist  = fig.add_subplot(gs[0, 1])
    ax_cmean = fig.add_subplot(gs[1, 0])
    ax_cvar  = fig.add_subplot(gs[1, 1])

    # --- Paths ---
    _apply_base_style(ax_paths, "Sample price paths")
    n_plot = min(25, paths.shape[0])
    for i in range(n_plot):
        ax_paths.plot(paths[i], color=BLUE if i == 0 else "#8AB8E6",
                      lw=1.4 if i == 0 else 0.7, alpha=0.65 if i == 0 else 0.25)
    ax_paths.axhline(S0, color=AMBER, lw=1.2, ls="--")
    ax_paths.set_xlabel("Time step", fontsize=9, color=GRAY)
    ax_paths.set_ylabel("Price ($)", fontsize=9, color=GRAY)

    # --- Distribution ---
    _apply_base_style(ax_dist, "Terminal price distribution")
    counts, edges, patches = ax_dist.hist(final_prices, bins=55, color=BLUE,
                                          alpha=0.75, edgecolor="white", lw=0.3)
    for patch, left in zip(patches, edges[:-1]):
        if left < var_95:
            patch.set_facecolor(RED); patch.set_alpha(0.8)
    ax_dist.axvline(var_95, color=RED, lw=1.4, label=f"VaR = {var_95:.1f}")
    ax_dist.axvline(cvar_95, color=CORAL, lw=1.4, ls=":", label=f"CVaR = {cvar_95:.1f}")
    ax_dist.legend(fontsize=8, framealpha=0.6)
    ax_dist.set_xlabel("Terminal price ($)", fontsize=9, color=GRAY)

    # --- Mean convergence ---
    _apply_base_style(ax_cmean, "E[S_T] convergence")
    ax_cmean.plot(conv["n"], conv["mean"], color=BLUE, lw=1.5, marker="o", ms=3)
    ax_cmean.axhline(analytical_mean, color=AMBER, lw=1.2, ls="--",
                     label=f"analytical {analytical_mean:.2f}")
    ax_cmean.legend(fontsize=8, framealpha=0.6)
    ax_cmean.set_xlabel("N", fontsize=9, color=GRAY)

    # --- VaR convergence ---
    _apply_base_style(ax_cvar, "VaR 95% convergence")
    ax_cvar.plot(conv["n"], conv["var_95"], color=RED, lw=1.5, marker="o", ms=3)
    ax_cvar.set_xlabel("N", fontsize=9, color=GRAY)

    fig.suptitle("GBM Monte Carlo — simulation dashboard",
                 fontsize=12, fontweight="normal", color="#2C2C2A", y=1.01)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig