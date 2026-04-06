# Monte Carlo Simulation for Financial Risk Estimation

## Objective
To analyze the effectiveness and convergence behavior of Monte Carlo simulations in financial modeling under varying volatility levels and simulation sizes.

## Methodology
We simulate stock price paths using Geometric Brownian Motion (GBM) with the stochastic differential equation:

$$ dS = \mu S dt + \sigma S dW $$

Where:
- $S$: Stock price
- $\mu$: Drift (expected return)
- $\sigma$: Volatility
- $dW$: Wiener process increment

Simulations are run with different numbers of paths and volatility levels to assess convergence and reliability of risk estimates.

## Parameters Used
- Initial price ($S_0$): $100
- Drift ($\mu$): $0.05$ (5% annual return)
- Time horizon ($T$): $1$ year
- Time step ($dt$): $0.01$
- Volatilities tested: $0.1, 0.2, 0.4$
- Simulation sizes: $100, 500, 1000, 5000$

## Results

### Convergence Analysis
The convergence plots show how statistical estimates stabilize as the number of simulations increases. This is crucial for determining the minimum number of simulations needed for reliable risk estimates.

### Distribution Analysis
Higher volatility leads to wider price distributions and increased risk measures.

### Key Findings
- Monte Carlo estimates become more stable with increasing simulation counts
- Value at Risk (VaR) converges more slowly than mean estimates
- Higher volatility amplifies the impact of simulation size on estimate reliability

## Conclusion
Monte Carlo simulation is an effective tool for financial risk estimation, but requires sufficient computational runs (typically >1000) for reliable results, especially under high volatility conditions.