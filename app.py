from flask import Flask, request, jsonify, render_template
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.linalg import cholesky
from scipy import optimize
from openai import OpenAI
import json




# Initialize OpenAI client with your API key
client = OpenAI(api_key="")

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')
# Function to format tickers using OpenAI
def format_tickers(raw_tickers):
    prompt = f"Format the following list of tickers into a standardized, comma-separated format without spaces: {raw_tickers}"
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt}
        ]
    )
    formatted_tickers = completion.choices[0].message.content.strip()
    return [ticker.strip().upper() for ticker in formatted_tickers.split(',') if ticker]

# Function to optimize portfolio weights using Sharpe Ratio
def MaximizeSharpeRatioOptmzn(mean_returns, cov_returns, risk_free_rate):
    def neg_sharpe_ratio(weights, mean_returns, cov_returns, risk_free_rate):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_returns, weights)))
        return -(portfolio_return - risk_free_rate) / portfolio_stddev

    num_assets = len(mean_returns)
    args = (mean_returns, cov_returns, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = num_assets * [1. / num_assets,]
    
    result = optimize.minimize(neg_sharpe_ratio, initial_weights, args=args,
                               method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

@app.route('/portfolioAnalysis', methods=['GET', 'POST'])
def portfolio_analysis():
    if request.method == 'POST':
        data = request.get_json()  # 获取 JSON 数据
        raw_tickers = data.get('tickers')
        if not raw_tickers:
            return "Tickers are required.", 405
    elif request.method == 'GET':
        raw_tickers = request.args.get('tickers')  # 使用 args 处理查询字符串参数
    if raw_tickers:
        tickers = format_tickers(raw_tickers)

        # Define the historical data period (last 60 years)
        end_date = '2024-01-01'
        start_date = f'{int(end_date[:4]) - 60}-01-01'

        # Download the historical data from Yahoo Finance
        data = yf.download(tickers, start=start_date, end=end_date, interval='1mo')['Adj Close']
        data = data.dropna()

        # Calculate log returns and covariance matrix
        log_returns = np.log(data / data.shift(1)).dropna()
        log_returns = log_returns.replace([np.inf, -np.inf], np.nan).dropna()
        cov_matrix = log_returns.cov()
        correlation_matrix_historical = log_returns.corr()

        # Simulate returns for the next 10 years (Cholesky and Eigenvalue)
        simulation_years = 10  # Number of years to simulate
        num_simulations = 1000  # Number of simulations per year

        cholesky_matrix = cholesky(cov_matrix, lower=True)
        random_normals_cholesky = np.random.normal(size=(simulation_years * num_simulations, len(tickers)))
        simulated_returns_cholesky = random_normals_cholesky @ cholesky_matrix.T
        simulated_returns_standardized_cholesky = (simulated_returns_cholesky - simulated_returns_cholesky.mean(axis=0)) / simulated_returns_cholesky.std(axis=0)
        correlation_matrix_cholesky = np.corrcoef(simulated_returns_standardized_cholesky, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        transformation_matrix_eigen = eigenvectors @ np.diag(np.sqrt(eigenvalues))
        random_normals_eigen = np.random.normal(size=(simulation_years * num_simulations, len(tickers)))
        simulated_returns_eigen = random_normals_eigen @ transformation_matrix_eigen.T
        simulated_returns_standardized_eigen = (simulated_returns_eigen - simulated_returns_eigen.mean(axis=0)) / simulated_returns_eigen.std(axis=0)
        correlation_matrix_eigen = np.corrcoef(simulated_returns_standardized_eigen, rowvar=False)

        # Calculate mean returns and covariance matrix for optimization (Cholesky)
        mean_returns_cholesky = np.mean(simulated_returns_cholesky, axis=0)
        cov_returns_cholesky = np.cov(simulated_returns_cholesky, rowvar=False)

        # Calculate mean returns and covariance matrix for optimization (Eigenvalue)
        mean_returns_eigen = np.mean(simulated_returns_eigen, axis=0)
        cov_returns_eigen = np.cov(simulated_returns_eigen, rowvar=False)

        # Fetch the 10-year US Treasury yield as the risk-free rate
        tnx_data = yf.download('^TNX', period='1d')
        tnx_yield_annual = tnx_data['Close'].iloc[-1] / 100
        risk_free_rate = (1 + tnx_yield_annual) ** (1 / 12) - 1

        # Calculate optimal weights for Cholesky and Eigenvalue
        optimal_weights_cholesky = MaximizeSharpeRatioOptmzn(mean_returns_cholesky, cov_returns_cholesky, risk_free_rate)
        optimal_weights_eigen = MaximizeSharpeRatioOptmzn(mean_returns_eigen, cov_returns_eigen, risk_free_rate)

        # Format the correlation matrices using pandas DataFrame
        corr_matrix_historical_df = pd.DataFrame(correlation_matrix_historical, index=tickers, columns=tickers)
        corr_matrix_cholesky_df = pd.DataFrame(correlation_matrix_cholesky, index=tickers, columns=tickers)
        corr_matrix_eigen_df = pd.DataFrame(correlation_matrix_eigen, index=tickers, columns=tickers)

        # Prepare the prompt for OpenAI explanation
        explanation_prompt = (
            f"The user has created a portfolio with the following tickers: {tickers}. "
            f"The optimized portfolio weights using the Cholesky method are: {optimal_weights_cholesky}. "
            f"Using the Eigenvalue method, the weights are: {optimal_weights_eigen}. "
            "Please analyze the provided correlation matrices from the historical data, "
            "Cholesky simulation, and Eigenvalue simulation. Consider current market conditions, "
            "recent news, and stock data to give a comprehensive analysis."
        )

        # Provide all the matrices and other information to OpenAI
        full_prompt = (
            f"Historical Correlation Matrix:\n{corr_matrix_historical_df}\n\n"
            f"Cholesky Simulation Correlation Matrix:\n{corr_matrix_cholesky_df}\n\n"
            f"Eigenvalue Simulation Correlation Matrix:\n{corr_matrix_eigen_df}\n\n"
            f"{explanation_prompt}"
        )

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": full_prompt}
            ]
        )

        # Respond with the optimized weights, correlation matrices, and explanation
        response = {
            "tickers": tickers,
            "optimized_weights_cholesky": optimal_weights_cholesky.tolist(),
            "optimized_weights_eigen": optimal_weights_eigen.tolist(),
            "correlation_matrix_historical": corr_matrix_historical_df.to_dict(),
            "correlation_matrix_cholesky": corr_matrix_cholesky_df.to_dict(),
            "correlation_matrix_eigen": corr_matrix_eigen_df.to_dict(),
            "explanation": completion.choices[0].message.content.strip()
        }

        return jsonify(response)
    else:
        return "Please provide tickers in the query string or in the request body.", 400

if __name__ == '__main__':
    app.run(debug=True)
