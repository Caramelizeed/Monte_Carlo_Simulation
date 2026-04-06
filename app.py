"""
Monte Carlo GBM Simulation Dashboard
Interactive web app for financial risk analysis using Streamlit.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from simulation import simulate_gbm, final_prices
from analytics import compute_all_stats, convergence_analysis, analytical_mean
from visualization import plot_price_paths, plot_distribution, plot_convergence

# Page configuration
st.set_page_config(
    page_title="Monte Carlo GBM Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<div class="main-header">📊 Monte Carlo GBM Simulator</div>', unsafe_allow_html=True)
    st.markdown("**Interactive Financial Risk Analysis using Geometric Brownian Motion**")

    # Sidebar for parameters
    st.sidebar.markdown('<div class="sidebar-header">⚙️ Simulation Parameters</div>', unsafe_allow_html=True)

    col1, col2 = st.sidebar.columns(2)

    with col1:
        S0 = st.slider("Initial Price (S₀)", 10.0, 500.0, 100.0, 10.0, help="Starting asset price")
        mu = st.slider("Drift (μ)", -0.2, 0.5, 0.10, 0.01, help="Annual expected return")
        T = st.slider("Time Horizon (T)", 0.25, 5.0, 1.0, 0.25, help="Years to simulate")

    with col2:
        sigma = st.slider("Volatility (σ)", 0.05, 1.0, 0.25, 0.01, help="Annual volatility")
        N = st.slider("Simulations (N)", 1000, 100000, 10000, 1000, help="Number of Monte Carlo paths")
        conf = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01, help="VaR/CVaR confidence")

    rf = st.sidebar.slider("Risk-Free Rate", 0.0, 0.10, 0.05, 0.01, help="For Sharpe ratio calculation")

    # Run simulation button
    if st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True):
        run_simulation(S0, mu, sigma, T, N, conf, rf)

def run_simulation(S0, mu, sigma, T, N, conf, rf):
    """Run the Monte Carlo simulation and display results."""

    with st.spinner("Running Monte Carlo simulation..."):
        # Simulate
        paths = simulate_gbm(S0=S0, mu=mu, sigma=sigma, T=T, dt=1/252, num_simulations=N, seed=42)
        finals = final_prices(paths)

        # Compute analytics
        stats = compute_all_stats(
            final_prices=finals, S0=S0, mu=mu, sigma=sigma, T=T,
            confidence=conf, rf=rf
        )
        conv = convergence_analysis(finals)

    # Display results
    st.success(f"✅ Simulation completed! {N:,} paths generated in {T} years.")

    # Key metrics in columns
    st.markdown("### 📈 Key Risk & Return Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Expected Price", f"${stats.mean_simulated:.2f}",
                 f"{(stats.mean_simulated/stats.mean_analytical-1)*100:.1f}% vs analytical")
        st.metric("Probability > S₀", f"{stats.prob_gain:.1%}")

    with col2:
        st.metric("VaR Loss", f"-${stats.var_95_loss:.2f}",
                 f"at {conf:.0%} confidence")
        st.metric("CVaR Loss", f"-${stats.cvar_95_loss:.2f}",
                 f"at {conf:.0%} confidence")

    with col3:
        st.metric("Volatility", f"${stats.std:.2f}")
        st.metric("Sharpe Ratio", f"{stats.sharpe_proxy:.3f}")

    with col4:
        st.metric("MC Bias", f"{(stats.mean_simulated/stats.mean_analytical-1)*100:.2f}%")
        st.metric("Median Price", f"${stats.median_simulated:.2f}")

    # Plots
    st.markdown("### 📊 Visual Analysis")

    tab1, tab2, tab3 = st.tabs(["Price Paths", "Distribution", "Convergence"])

    with tab1:
        st.markdown("**Sample Price Trajectories**")
        fig = plot_price_paths(paths, S0=S0, max_paths=20, show=False)
        st.pyplot(fig)
        plt.close(fig)

    with tab2:
        st.markdown("**Terminal Price Distribution**")
        fig = plot_distribution(finals, S0=S0, var_95=stats.var_95,
                               cvar_95=stats.cvar_95, show=False)
        st.pyplot(fig)
        plt.close(fig)

    with tab3:
        st.markdown("**Monte Carlo Convergence**")
        analytical_mean_val = analytical_mean(S0, mu, T)
        fig = plot_convergence(conv, analytical_mean=analytical_mean_val, show=False)
        st.pyplot(fig)
        plt.close(fig)

    # Detailed statistics table
    st.markdown("### 📋 Detailed Statistics")

    with st.expander("View Complete Statistics Table"):
        stats_dict = {
            "Metric": [
                "Analytical Mean", "Simulated Mean", "Analytical Median", "Simulated Median",
                "Standard Deviation", "Variance", f"VaR {conf:.0%} (Price)", f"VaR {conf:.0%} (Loss)",
                f"CVaR {conf:.0%} (Price)", f"CVaR {conf:.0%} (Loss)", "P(S_T > S₀)", "Sharpe Proxy"
            ],
            "Value": [
                f"${stats.mean_analytical:.4f}", f"${stats.mean_simulated:.4f}",
                f"${stats.median_analytical:.4f}", f"${stats.median_simulated:.4f}",
                f"${stats.std:.4f}", f"${stats.variance:.4f}",
                f"${stats.var_95:.4f}", f"-${stats.var_95_loss:.4f}",
                f"${stats.cvar_95:.4f}", f"-${stats.cvar_95_loss:.4f}",
                f"{stats.prob_gain:.2%}", f"{stats.sharpe_proxy:.4f}"
            ]
        }
        st.table(stats_dict)

if __name__ == "__main__":
    main()