# Portfolio Optimizer 

A Python-based tool for analyzing and optimizing investment portfolios using historical market data.

The project focuses on applying quantitative methods to explore risk–return trade-offs and portfolio allocation strategies.

**Live app:** https://portfolio-optimizer-btynevtuy9f3fje8xq35vv.streamlit.app/
---

## Overview

This project implements basic portfolio optimization techniques to help evaluate how different asset allocations affect portfolio performance.

It is intended as an educational and exploratory tool rather than a production trading system.

---

## Features

- Load and process historical price data
- Compute portfolio returns and volatility
- Optimize asset weights under common constraints
- Evaluate risk–return trade-offs
- Visualize portfolio performance

---

## Tech stack

- Python
- NumPy
- Pandas
- SciPy
- Matplotlib

---

## Running locally

```bash
git clone https://github.com/kina108/portfolio-optimizer.git
cd portfolio-optimizer

python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\Activate.ps1  # Windows

pip install -r requirements.txt
python main.py
