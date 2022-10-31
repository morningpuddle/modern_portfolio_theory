import yfinance as yf
import streamlit as st
import datetime 
import pandas as pd
import numpy as np
import requests
import math
import altair as alt
yf.pdr_override()

st.title('Modern Portfolio Theory')

expander = st.expander("What is Modern Portfolio Theory?")
expander.write("""
	Modern Portfolio Theory refers to an investment theory that allows investors to
	assemble an asset portfolio that **maximizes** expected
	**return** for a given level of **risk**.
""")

def user_input_features():
	st.sidebar.caption("Pick two stocks you want to find an asset portfolio that **maximizes** expected *return* for a given level of **risk**.")
	col1, col2 = st.sidebar.columns(2)
	ticker1 = col1.text_input("Ticker A", "MCD")
	ticker2 = col2.text_input("Ticker B", "AMZN")
	st.sidebar.caption("Pick market index and risk-free rate (U.S. Treasury)")
	market_ticker = st.sidebar.selectbox('Market Data',('SPY', 'VOO', 'VTI', 'VIOV', 'IWN'))
	rfr = st.sidebar.slider("Risk-Free Rate (%)", 0.00, 100.00, 3.00)

	st.sidebar.caption("Pick start/end date for tickers.")
	today = datetime.date.today()
	tenyrs_ago = datetime.date.today() - datetime.timedelta(days=3652)
	start = pd.to_datetime(st.sidebar.date_input("Start Date", tenyrs_ago))
	end = pd.to_datetime(st.sidebar.date_input("End Date", today))

	return ticker1, ticker2, market_ticker, start, end, rfr

@st.cache
def download(ticker1, ticker2, market_ticker):
	return yf.download(ticker1 + " " + ticker2 + " " + market_ticker,start,end)

ticker1, ticker2, market_ticker, start, end, rfr = user_input_features()

# preliminary check if ticker exists
nonexistant = []
for t in [ticker1, ticker2, market_ticker]:
	info = yf.Ticker(t).info
	if info['regularMarketPrice'] == None:
		nonexistant += [t]
if len(nonexistant) > 0:
	e = RuntimeError("Following tickers don\'t exist: " + ", ".join(nonexistant))
	st.error(e)
	st.stop()
stocks = download(ticker1, ticker2, market_ticker)

st.header("Closing Prices 📈")
st.caption("Below shows daily closing prices for given duration from " + start.strftime("%Y-%m-%d") + " to " + end.strftime("%Y-%m-%d"))
prices = stocks["Close"]
prices_long_form = prices.reset_index()
prices_long_form = prices_long_form.rename(columns={'index':'Date'})
prices_long_form = prices_long_form.melt("Date", var_name="ticker", value_name="price")
hover = alt.selection_single(
    fields=["Date"],
    nearest=True,
    on="mouseover",
    empty="none",
)
st.altair_chart(
	alt.Chart(prices_long_form)
	.mark_line(interpolate='step-after').encode(
		x="Date", 
		y="price", 
		color="ticker",
		strokeDash="ticker",
		tooltip=[
				alt.Tooltip("ticker", title="Ticker"),
                alt.Tooltip("Date", title="Date"),
                alt.Tooltip("price", title="Price (USD)"),
         ]).add_selection(hover),
	use_container_width=True)

def summary_stat_each(ticker, data, market_return_pct, col):
	closing_price = data
	daily_return = closing_price.pct_change().dropna()
	er_daily = daily_return.mean()
	er_yearly =((1+er_daily)**252)-1
	risk_daily = daily_return.std()
	risk_yearly = risk_daily*math.sqrt(12)
	beta = market_return_pct
	return daily_return, er_daily, er_yearly, risk_daily,risk_yearly

st.header("Summary Statistics 📝")
expander = st.expander("What does these statistics tell me?")
expander.write("""
	Modern Portfolio Theory employes the core idea of *diversification* (holding assets that are less correlated).
	Diversification in a portfolio allocation strategy aims to minimize unsystematic risks (sepcific to each asset/stock).

	* **Correlation** tells how correlated the two specified stock/assets are. The higher the number, the more correlated
	* **Expected daily return** is the average(percentage diff of day N and day N + 1).
	* **Expected yearly return** is computed by (1 + expected daily return)^252, where 252 is the number of trading days in a year.
	* **Daily Risk** is the standard deviation of daily return.
	* **Yearly Risk** is the standard deviation of yearly return.
""")
data1= prices[ticker1]
data2= prices[ticker2]
market_data = prices[market_ticker]
market_return_pct = market_data.pct_change().dropna()
col1, col2 = st.columns(2)
daily_return1,er_daily1,er_yearly1,risk_daily1,risk_yearly1 = summary_stat_each(ticker1, data1, market_return_pct, col1)
daily_return2,er_daily2,er_yearly2,risk_daily2,risk_yearly2 = summary_stat_each(ticker2, data2, market_return_pct, col2)
df = pd.DataFrame(
	data=np.array([
		['{:.2f}%'.format(er_daily1*100),'{:.2f}%'.format(er_daily2*100)],
		['{:.2f}%'.format(er_yearly1*100),'{:.2f}%'.format(er_yearly2*100)],
		['{:.2f}%'.format(risk_daily1*100),'{:.2f}%'.format(risk_daily2*100)],
		['{:.2f}%'.format(risk_yearly1*100),'{:.2f}%'.format(risk_yearly2*100)],
		]),
	columns=[ticker1, ticker2],
	index=['Expected Return (daily)', 'Expected Return (yearly)', 'Risk (daily)', 'Risk (yearly)']
)

stacked = np.vstack((daily_return2, daily_return1)).T
correlation_daily = np.corrcoef(daily_return2, daily_return1)[0][1]
st.metric(label="Correlation Factor", value="{:.2f}".format(correlation_daily))
st.table(df)
covariance_yearly = np.cov(daily_return2, daily_return1)*math.sqrt(252.0)

# st.caption("TODO: following seem off from excel")
# st.write(correlation_daily)
# st.write(covariance_yearly)
# col1, col2 = st.columns(2)
# col1.metric("Correlation(daily)", '{:.2f}'.format(correlation_daily))
# col2.metric("Covariance(yearly)", '{:.2f}'.format(covariance_yearly))

st.header("\nIdeal Diversification ⚖️")
expander = st.expander("How do I find the ideal diversification ratio between two stocks/assets?")
expander.write("""
	We compute two things - 1. Capital Allocation Line and 2. Efficient Portfolio

	* **Capital Allocation Line** is the projected **return** & **risk** computed based on different ratio of (max return given by two assets) and (risk free rate).
	* **Efficient Portfolio** is the projected **return** & **risk** computed based on different ratio of (asset 1) and (asset 2).
	* **Return** of two stocks are computed by `(ratio of asset 1) * yearly return of asset 1 + (ratio of asset 2) * yearly return of asset 2` by where ratio sums up to 1.
	* **Risk** of two stocks are computed by weight, daily expected returns, and correlation. 
""")

import matplotlib.pyplot as plt
weights1 = np.arange(1.30, -0.3, -0.02)
weights2 = 1 - weights1
portfolio_risk = np.sqrt(weights1**2 * risk_daily1**2 + weights2**2 * risk_daily2**2 + 2 * weights1 * weights2 * correlation_daily * risk_daily1 * risk_daily2)
portfolio_return = weights1 * er_yearly1 + weights2 * er_yearly2
data = np.vstack((weights1, portfolio_risk, portfolio_return)).T

# ideal portfolio
sharpe = (portfolio_return - rfr/100.0) /portfolio_risk
max_idx = np.argmax(sharpe)
max_sharpe = sharpe[max_idx]
max_return = portfolio_return[max_idx]
max_risk   = portfolio_risk[max_idx]
cal_x = np.array([0, 1.0, 2.0])
cal_return = cal_x * max_return + (1 - cal_x) * rfr/100.0
cal_risk = cal_x * max_risk
cal = np.column_stack((cal_x, cal_risk, cal_return))

chart_data = pd.DataFrame(
    data,
    columns=['Weight of ' + ticker1, 'Risk (Std)', 'Return %'])
cal_data = pd.DataFrame(
	cal,
	columns = ['Weight of Optimal', 'Risk (Std)', 'Return %'])

st.caption("Optimal Ratio of " + ticker1 + ":" + ticker2 +  " is where Capital Allocation Line and Efficient Portfolio intersects in tangential fashion")

st.altair_chart(alt.layer(
	alt.Chart(chart_data).mark_circle(size=30).encode(
		x='Risk (Std)',
    	y='Return %',
    	tooltip=['Weight of ' + ticker1, 'Risk (Std)', 'Return %']),
	alt.Chart(cal_data).mark_line().encode(
		x='Risk (Std)',
		y='Return %',
		tooltip=['Weight of Optimal', 'Risk (Std)', 'Return %']
)),
use_container_width=True)

# Solve for exact ratio
# https://stackoverflow.com/questions/67379860/finding-intersection-of-two-functions-in-python-scipy-root-finding
from scipy.optimize import root
def g(x):
	res = math.sqrt((x - er_yearly1)**2+risk_daily1**2 + (x - er_yearly2)**2+risk_daily2**2 + 2 * risk_daily1 * risk_daily2 * correlation_daily * (x - er_yearly1) * (x - er_yearly2)) / (er_yearly1 - er_yearly2) - max_risk * (x - rfr/100.0)/(max_return - rfr/100.0)
	return res

sol = root(g, [0], method='hybr')
max_return_found = sol.x
st.metric(label="Max Return of Optimal Portfolio", value="{:.2f}%".format(max_return_found[0]* 100))

# return of optimal = w1 * yearly return 1 + (1-w1) * yearly return 2
# which equals to 
# w1 = (return - yearly return 2) / (yearly return 1 - yearly return 2)
w1 = (max_return_found - er_yearly2) / (er_yearly1 - er_yearly2)

ideal_df = pd.DataFrame(
	data=np.array([
		['{:.2f}%'.format(w1[0]*100),'{:.2f}%'.format((1-w1[0])*100)]
		]),
	columns=[ticker1, ticker2],
	index=['Ideal Weight']
)

st.table(ideal_df)