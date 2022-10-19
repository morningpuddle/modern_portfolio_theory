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

# TODO: Latex equation of CAPM

# st.subheader("Part 1: CAPM (Capital Asset Pricing Model)")
# if st.checkbox("Show"):

# 	# TODO: ticker (SPY, etc) with st.metric!
# 	mr = st.number_input("Market Return (yearly) %", 0.00, 100.00, 20.00, format="%f")

# 	beta_range = np.arange(0, 2, 0.1)
# 	return_val = beta_range * (mr - rfr) + rfr
# 	data = np.vstack((beta_range, return_val)).T
# 	# st.write(data)
# 	chart_data = pd.DataFrame(
# 	    data,
# 	    columns=['beta', 'Return%'])
# 	st.line_chart(chart_data, x="beta", y="Return%")



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
ticker1, ticker2, market_ticker, start, end,rfr = user_input_features()
stocks = download(ticker1, ticker2, market_ticker)

st.header("Closing Prices 📈")
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
	col.text(ticker)
	col.text('Expected Return (daily): {:.2f}%'.format(er_daily*100))
	col.text('Expected Return (yearly): {:.2f}%'.format(er_yearly*100))
	col.text('Risk (daily): {:.2f}%'.format(risk_daily*100))
	col.text('Risk (yearly): {:.2f}%'.format(risk_yearly*100))
	return daily_return,er_yearly,risk_yearly

st.header("Summary Stats 📝")
data1= prices[ticker1]
data2= prices[ticker2]
market_data = prices[market_ticker]
market_return_pct = market_data.pct_change().dropna()
col1, col2 = st.columns(2)
daily_return1,er_yearly1,risk_daily1 = summary_stat_each(ticker1, data1, market_return_pct, col1)
daily_return2,er_yearly2,risk_daily2 = summary_stat_each(ticker2, data2, market_return_pct, col2)

stacked = np.vstack((daily_return2, daily_return1)).T
correlation_daily = np.corrcoef(daily_return2, daily_return1)[0][1]
st.text("correlation_daily: {:.2f}".format(correlation_daily))
covariance_yearly = np.cov(daily_return2, daily_return1)*math.sqrt(252.0)
# st.caption("TODO: following seem off from excel")
# st.write(correlation_daily)
# st.write(covariance_yearly)

# col1, col2 = st.columns(2)
# col1.metric("Correlation(daily)", '{:.2f}'.format(correlation_daily))
# col2.metric("Covariance(yearly)", '{:.2f}'.format(covariance_yearly))

st.header("\nIdeal Diversification ⚖️")
import matplotlib.pyplot as plt
weights1 = np.arange(1.30, -0.3, -0.05)
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

st.altair_chart(alt.layer(alt.Chart(chart_data).mark_circle(size=30).encode(
		x='Risk (Std)',
    	y='Return %',
    	tooltip=['Weight of ' + ticker1, 'Risk (Std)', 'Return %']),
alt.Chart(cal_data).mark_line().encode(
		x='Risk (Std)',
		y='Return %',
		tooltip=['Weight of Optimal', 'Risk (Std)', 'Return %']
)),
use_container_width=True)


# DATE_COLUMN = 'date/time'
# DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
#          'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

# @st.cache
# def load_data(nrows):
#     data = pd.read_csv(DATA_URL, nrows=nrows)
#     lowercase = lambda x: str(x).lower()
#     data.rename(lowercase, axis='columns', inplace=True)
#     data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
#     return data

# # Create a text element and let the reader know the data is loading.
# data_load_state = st.text('Loading data...')
# # Load 10,000 rows of data into the dataframe.
# data = load_data(10000)
# # Notify the reader that the data was successfully loaded.
# data_load_state.text("Done! (using st.cache)")

# if st.checkbox("Show raw data"):
# 	st.subheader('Raw data')
# 	st.write(data)

# st.subheader('Number of pickups by hour')
# hist_values = np.histogram(
#     data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]

# st.bar_chart(hist_values)

# hour_to_filter = st.slider('hour', 0, 23, 17)  # min: 0h, max: 23h, default: 17h
# #hour_to_filter = 17
# filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
# st.subheader(f'Map of all pickups at {hour_to_filter}:00')
# st.map(filtered_data)
