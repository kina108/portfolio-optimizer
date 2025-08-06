import streamlit as st
from portfolio import optimize_portfolio, plot_efficient_frontier

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
st.title("📈 Portfolio Optimizer")

tickers_input = st.text_input(
    "Enter asset tickers (comma-separated):", "AAPL,MSFT,GOOGL"
)

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
strategy = st.selectbox(
    "Optimization Strategy:", ["min_variance", "max_sharpe_ratio"]
)

if st.button("Optimize Portfolio"):
    weights, stats = optimize_portfolio(tickers, strategy)
    st.subheader("Optimal Weights")
    st.write(dict(zip(tickers, weights)))
    st.subheader("Portfolio Stats")
    st.write(stats)
    fig = plot_efficient_frontier(tickers, strategy)
    st.pyplot(fig)


