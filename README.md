# Beyond Forecasts: Deep Learning for Direct Portfolio Optimization

This repository contains the implementation of a deep learning model for portfolio optimization, based on the research paper **"Deep Learning for Portfolio Optimization"** by Zhang, Zohren, and Roberts (Oxford, 2021). Our project moves beyond traditional forecasting-based methods by training a Long Short-Term Memory (LSTM) network to **directly maximize the portfolio's Sharpe Ratio**.

## Table of Contents
- [1. The Vision: A New Path for Portfolio Optimization](#1-the-vision-a-new-path-for-portfolio-optimization)
- [2. Core Concept: End-to-End Learning](#2-core-concept-end-to-end-learning)
- [3. Methodology](#3-methodology)
  - [3.1. Smart Asset Selection](#31-smart-asset-selection)
  - [3.2. Feature Engineering & Input Preparation](#32-feature-engineering--input-preparation)
  - [3.3. Model Architecture: The LSTM Brain](#33-model-architecture-the-lstm-brain)
  - [3.4. The Secret Sauce: Sharpe Ratio as a Loss Function](#34-the-secret-sauce-sharpe-ratio-as-a-loss-function)
- [4. Training & Validation](#4-training--validation)
  - [4.1. Walk-Forward Validation](#41-walk-forward-validation)
  - [4.2. Benchmark Strategies](#42-benchmark-strategies)
- [5. Performance & Results](#5-performance--results)
- [6. Technology Stack](#6-technology-stack)
- [7. How to Run This Project](#7-how-to-run-this-project)
- [8. Conclusion & Future Work](#8-conclusion--future-work)
- [9. Acknowledgements](#9-acknowledgements)

## 1. The Vision: A New Path for Portfolio Optimization

For decades, portfolio optimization has followed a two-step process:
1.  **Forecast:** Predict the expected returns and covariances of assets.
2.  **Optimize:** Use these forecasts (e.g., with Modern Portfolio Theory) to find the "optimal" portfolio weights.

This approach has a fundamental weakness: **forecasting financial markets is incredibly difficult and often unreliable.**

Our project explores a different paradigm: **What if a model could learn an optimal allocation strategy directly from market history, bypassing the need for explicit forecasts?**

Our vision was to build and test a deep learning system that learns to allocate assets with the specific, end-to-end goal of maximizing risk-adjusted performance, as measured by the Sharpe Ratio.

## 2. Core Concept: End-to-End Learning

The core idea of our strategy is to create a direct link between market data and portfolio performance.

Instead of training a model to minimize prediction error (e.g., MSE on future prices), we train it to **minimize the negative Sharpe Ratio**. This aligns the model's training objective directly with the final goal of a portfolio manager: achieving the best possible return for a given level of risk.


*Figure: The model takes concatenated asset features as input and directly outputs portfolio weights via a softmax layer. The final performance (Rp) is used to calculate the Sharpe Ratio for backpropagation.*

## 3. Methodology

### 3.1. Smart Asset Selection
Instead of grappling with thousands of individual stocks, we constructed our portfolio from a curated set of four highly diversified Exchange-Traded Funds (ETFs). This provides broad exposure across key market segments while keeping the problem tractable.

-   **VTI (Stocks):** Vanguard Total Stock Market ETF, capturing the broad US equity market.
-   **AGG (Bonds):** iShares Core U.S. Aggregate Bond ETF, representing the stability of the US bond market.
-   **DBC (Commodities):** Invesco DB Commodity Index Tracking Fund, offering diversification through raw materials.
-   **VIX (Volatility):** While we use VIX data, a tradable equivalent like VXX or VIXY would be used to hedge or capture market stress.

### 3.2. Feature Engineering & Input Preparation
The LSTM model is fed a rich, sequential context to make its decisions.

-   **Rolling Window:** We use a rolling window of the **past 50 trading days** to form a single input sequence.
-   **Input Features:** For each day in the window, we provide the model with the **price and return data** for all four assets.
-   **Data Structure:** This process results in a 3D dataset `(samples, timesteps, features)` perfectly formatted for an LSTM. `timesteps` is 50, and `features` is 8 (4 assets * 2 features).

### 3.3. Model Architecture: The LSTM Brain
Our network is designed to effectively process and learn from sequential financial data.

1.  **Input Layer:** Receives the `(50, 8)` input sequences.
2.  **Stacked LSTM Layers:** Two stacked LSTM layers (64 units followed by 32 units) process the temporal patterns and capture long-term dependencies in the data.
3.  **Dropout Layer:** A Dropout layer is used between the LSTMs for regularization, preventing the model from simply memorizing the training data (overfitting).
4.  **Dense Output Layer:** A final Dense layer with **4 neurons** (one for each asset).
5.  **Softmax Activation:** This crucial activation function converts the layer's raw outputs into a probability distribution. The result is a set of portfolio weights that are all positive and sum to 100%, representing a fully invested, long-only portfolio.

### 3.4. The Secret Sauce: Sharpe Ratio as a Loss Function
This is the most critical component of our project. We implemented a custom loss function that calculates the **negative Sharpe Ratio** of the portfolio's returns over a training batch.

`loss = - (mean_return / std_dev_return)`

The Adam optimizer's goal is to minimize this value. By minimizing the negative Sharpe Ratio, the network is implicitly guided, via gradient ascent, to find weights that **maximize the actual Sharpe Ratio**.

## 4. Training & Validation

### 4.1. Walk-Forward Validation
To realistically evaluate the strategy and avoid lookahead bias, we employed a rigorous **walk-forward validation** scheme. The model is trained and tested on incrementally advancing slices of time:
-   Train: 2006-2010 -> Test: 2011
-   Train: 2006-2011 -> Test: 2012
-   Train: 2006-2012 -> Test: 2013
-   ...and so on, up to 2020.

### 4.2. Benchmark Strategies
We compared our Deep Learning Strategy (DLS) against several traditional benchmarks:
-   **Fixed Allocation Strategies:** Four different static-weight portfolios (e.g., 60/40, equal weight).
-   **Mean-Variance Optimization (MVO):** The classic Markowitz model, updated daily with a 50-day rolling window.
-   **Maximum Diversification (MD):** A portfolio that aims to maximize diversification rather than just balancing risk and return.

## 5. Performance & Results
Our end-to-end LSTM model demonstrated superior performance in our testing framework.

-   **Key Result:** When evaluated on the annualized Sharpe Ratio across all walk-forward periods, the LSTM model **outperformed all fixed allocation and traditional MVO strategies**.
-   **Consistency:** Cumulative return plots show that the DLS strategy achieved more robust long-term growth compared to the benchmarks, successfully navigating market shifts, including the 2020 COVID-19 crisis.


*Figure: The LSTM strategy achieved the highest average Sharpe and Sortino Ratios compared to benchmarks.*

## 6. Technology Stack
-   **Language:** Python
-   **Libraries:**
    -   `TensorFlow` / `Keras` for building and training the LSTM model.
    -   `Pandas` for data manipulation and time-series handling.
    -   `NumPy` for numerical operations.
    -   `Scikit-learn` for data preprocessing.
    -   `Matplotlib` / `Seaborn` for plotting results.

## 7. How to Run This Project
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/dl-portfolio-optimization.git
    cd dl-portfolio-optimization
    ```
2.  **Set up a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```
3.  **Download Data:** Place your historical ETF data (`.csv` format) in the `data/` directory.
4.  **Run the Backtest:**
    ```bash
    python main.py
    ```

## 8. Conclusion & Future Work
This project successfully demonstrates that an end-to-end deep learning approach, optimized directly on a risk-adjusted performance metric, can be a powerful alternative to traditional portfolio optimization methods. Our LSTM model learned to effectively generate high-performing portfolios by directly interpreting market data patterns.

**Future Work:**
-   **Transaction Costs:** Incorporate transaction costs into the loss function to penalize high turnover.
-   **Alternative Risk Metrics:** Optimize for other metrics like the Sortino Ratio, Calmar Ratio, or Maximum Drawdown.
-   **Richer Data:** Integrate alternative data sources, such as macroeconomic indicators or sentiment analysis.
-   **Advanced Architectures:** Experiment with other deep learning models like Transformers, which may capture market dynamics differently.

## 9. Acknowledgements
This project is heavily inspired by and builds upon the foundational work presented in the following paper. We are grateful to the authors for their insightful contribution to the field.

> Zhang, Z., Zohren, S., & Roberts, S. (2021). *Deep Learning for Portfolio Optimization*. arXiv preprint arXiv:2005.13665.
