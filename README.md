# 📊 Monte Carlo Simulation for Financial Risk Estimation

A professional-grade Monte Carlo simulation framework for analyzing Geometric Brownian Motion (GBM) in quantitative finance. This tool provides comprehensive risk and return analysis with analytical benchmarks, convergence diagnostics, and publication-quality visualizations.

## 🎯 Core Research Question

**How does Monte Carlo simulation reliability vary with volatility levels and sample sizes for financial risk estimation?**

## ✨ Features

- **🔬 Exact GBM Simulation**: Vectorized Geometric Brownian Motion with log-normal discretization
- **📈 Comprehensive Risk Metrics**: VaR, CVaR, Sharpe ratio, probability of gain, and analytical benchmarks
- **📊 Convergence Analysis**: Quantifies Monte Carlo estimation stability across simulation counts
- **🎨 Professional Visualizations**: Publication-quality plots with custom styling
- **🖥️ Interactive Web App**: Streamlit-based GUI for easy parameter adjustment
- **💻 CLI Interface**: Flexible command-line parameters for batch analysis
- **✅ Analytical Validation**: Closed-form GBM solutions for accuracy verification

## 🚀 Quick Start

### 🌐 Web App (Interactive GUI - Recommended)

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the web app
python -m streamlit run app.py
```

Then open your browser to `http://localhost:8501`

**Web App Features:**
- Interactive sliders for all parameters (S₀, μ, σ, T, N, confidence, etc.)
- Real-time results and plots
- No terminal commands needed
- Professional dashboard with tabs for different visualizations

### 💻 Command Line (Advanced)

```bash
# Basic run with defaults
python experiments/run_experiments.py

# Custom high-volatility scenario
python experiments/run_experiments.py --S0 150 --mu 0.12 --sigma 0.40 --T 2 --N 50000

# Extreme risk analysis (99% VaR)
python experiments/run_experiments.py --conf 0.99 --sigma 0.35 --N 100000
```

## 📋 Requirements

- Python 3.9+
- NumPy >= 1.26
- Matplotlib >= 3.8
- Streamlit >= 1.28

## 🎮 Usage Examples

### Web App Interface

1. **Adjust Parameters**: Use sliders in the sidebar
   - Initial Price (S₀): $10 - $500
   - Drift (μ): -20% to 50%
   - Volatility (σ): 5% to 100%
   - Time Horizon (T): 0.25 to 5 years
   - Simulations (N): 1,000 to 100,000
   - Confidence Level: 90% to 99%

2. **Run Simulation**: Click "🚀 Run Simulation"

3. **Explore Results**:
   - **Key Metrics**: Expected price, VaR loss, Sharpe ratio
   - **Price Paths Tab**: Sample GBM trajectories
   - **Distribution Tab**: Terminal price histogram with risk overlays
   - **Convergence Tab**: Monte Carlo stability analysis

### Command Line Interface

```bash
# Conservative investment scenario
python experiments/run_experiments.py --S0 100 --mu 0.05 --sigma 0.15 --T 1 --N 25000

# Tech stock scenario
python experiments/run_experiments.py --S0 200 --mu 0.15 --sigma 0.45 --T 2 --N 50000

# Custom output directory
python experiments/run_experiments.py --outdir my_analysis --conf 0.99
```

## 📊 Output & Results

### Generated Files (CLI)
- `output/price_paths.png` - Sample simulated price trajectories
- `output/distribution.png` - Terminal price histogram with VaR/CVaR overlays
- `output/convergence.png` - Monte Carlo convergence diagnostics
- `output/dashboard.png` - Summary dashboard

### Key Metrics Explained

| Metric | Description |
|--------|-------------|
| **VaR (Value at Risk)** | Maximum expected loss at given confidence level |
| **CVaR (Conditional VaR)** | Expected loss in the worst-case scenarios |
| **Sharpe Ratio** | Risk-adjusted return measure (μ - rf) / σ |
| **Convergence** | How estimates stabilize with more simulations |
| **MC Bias** | Difference between analytical and simulated results |

### Sample Output

```
GBM MONTE CARLO — SIMULATION REPORT
──────────────────────────────────────────────────────
  Parameter                           Value
──────────────────────────────────────────────────────
  S₀ — initial price                $100.00
  μ  — drift (ann.)                  0.1000
  σ  — volatility (ann.)             0.2500
  T  — horizon (years)                 1.00
  N  — simulations                  10,000
  Confidence level                      95%

RISK & RETURN METRICS
──────────────────────────────────────────────────────
  E[S_T] analytical                 $110.5171
  E[S_T] simulated                  $110.4234
  VaR 95% (dollar loss)             -$34.5679
  CVaR 95% (dollar loss)            -$41.8766
  Sharpe proxy                         0.2000
──────────────────────────────────────────────────────
```

## 🏗️ Project Structure

```
monte_carlo_simulation/
├── 📄 README.md              # This file
├── 📋 requirements.txt       # Python dependencies
├── 🌐 app.py                 # Streamlit web application
├── 📊 experiments/
│   └── run_experiments.py    # CLI entry point
├── 📈 notebooks/
│   └── research.ipynb        # Jupyter analysis notebook
├── 📁 results/               # Generated plots and data
├── 📝 report/
│   └── findings.md           # Research summary
└── 🔧 src/
    ├── simulation.py         # GBM simulation engine
    ├── analytics.py          # Risk metrics & analytics
    └── visualization.py      # Plotting functions
```

## 🔬 Technical Details

### Mathematical Foundation

The simulation uses Geometric Brownian Motion:

$$ dS = \mu S dt + \sigma S dW $$

With exact log-normal discretization:

$$ S_{t+dt} = S_t \cdot \exp\left((\mu - \frac{1}{2}\sigma^2)dt + \sigma\sqrt{dt}\cdot Z\right)$$

Where $Z \sim \mathcal{N}(0,1)$

### Risk Measures

- **VaR**: $\inf\{x \in \mathbb{R} : P(S_T \leq x) > 1 - \alpha\}$
- **CVaR**: $\mathbb{E}[S_T | S_T \leq VaR_\alpha]$
- **Analytical Solutions**: Closed-form GBM moments for validation

## 🎓 Educational Value

This project demonstrates:
- **Stochastic Processes**: GBM and Wiener processes
- **Monte Carlo Methods**: Convergence and variance reduction
- **Risk Management**: VaR, CVaR, and tail risk analysis
- **Quantitative Finance**: Real-world financial modeling
- **Python Best Practices**: Modular design, type hints, documentation

## 🚀 Advanced Usage

### Batch Analysis
```bash
# Run multiple scenarios
for sigma in 0.1 0.2 0.3 0.4; do
    python experiments/run_experiments.py --sigma $sigma --N 50000
done
```

### Integration with Other Tools
```python
from src.simulation import simulate_gbm
from src.analytics import calculate_var

# Custom analysis
paths = simulate_gbm(S0=100, mu=0.1, sigma=0.25, T=1, dt=1/252, num_simulations=10000)
final_prices = paths[:, -1]
var_95 = calculate_var(final_prices, confidence=0.95)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure documentation is updated
5. Submit a pull request

## 📜 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Built with NumPy, Matplotlib, and Streamlit
- Inspired by quantitative finance literature
- Designed for educational and research purposes

---

**Ready to explore quantitative finance? Launch the web app and start simulating!** 🚀📊
