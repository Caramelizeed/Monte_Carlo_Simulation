# 🎲 Monte Carlo Simulation in Quantitative Finance

This repository explores **Monte Carlo Simulation**, one of the most powerful techniques used in quantitative finance for modeling uncertainty and predicting outcomes.

---

## 🚀 What is Monte Carlo Simulation?

Monte Carlo Simulation is a computational method that uses **random sampling** to estimate numerical results.

Instead of solving problems analytically, we:

1. Define a probabilistic model
2. Generate random inputs
3. Run simulations multiple times
4. Analyze the distribution of outcomes

---

## 📊 Why is it Important in Finance?

Financial markets are inherently uncertain. Monte Carlo methods help in:

* 📈 Stock price simulation
* 📉 Risk analysis
* 💰 Option pricing
* 📊 Portfolio optimization
* ⚠️ Value at Risk (VaR) estimation

---

## 🧠 Simple Intuition

Imagine trying to predict the future price of a stock.

Instead of guessing one value, we:

* Simulate **thousands of possible price paths**
* Each path represents a possible future
* Then analyze all outcomes statistically

---

## 🔢 Mathematical Idea

A basic Monte Carlo simulation for stock prices is often based on:

S(t) = S₀ · exp((μ - ½σ²)t + σW)

Where:

* S(t) → future stock price
* S₀ → initial price
* μ → expected return
* σ → volatility
* W → random variable (Brownian motion)

---

## ⚙️ Basic Python Example

```python
import numpy as np
import matplotlib.pyplot as plt

S0 = 100     # initial price
mu = 0.1     # expected return
sigma = 0.2  # volatility
T = 1        # time (1 year)
N = 252      # steps
simulations = 100

dt = T / N

paths = []

for _ in range(simulations):
    prices = [S0]
    for _ in range(N):
        random_shock = np.random.normal(0, 1)
        price = prices[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * random_shock)
        prices.append(price)
    paths.append(prices)

for path in paths:
    plt.plot(path)

plt.title("Monte Carlo Simulation of Stock Prices")
plt.xlabel("Time Steps")
plt.ylabel("Price")
plt.show()
```

---

## 🎥 Visual Explanation

---

## 📌 Key Takeaways

* Monte Carlo = **simulate many futures instead of predicting one**
* Works well when systems are **too complex for closed-form solutions**
* Widely used in **trading, risk management, and derivatives pricing**

---

## 🎯 Future Work

* 📌 Option pricing using Monte Carlo
* 📌 Comparing Monte Carlo vs Black-Scholes
* 📌 Portfolio risk simulations
* 📌 GPU-accelerated simulations

---

## 🤝 Contributions

Open to discussions on:

* Quant finance
* Simulation techniques
* Trading strategies

---
