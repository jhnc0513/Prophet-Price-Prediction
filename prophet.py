import pandas as pd
import prophet.plot
import yfinance as yf
from datetime import datetime
from datetime import timedelta
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import warnings
import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')
pd.options.display.float_format = '${:,.2f}'.format

today =datetime.today().strftime('%Y-%m-%d')
start_date = '2016-01-01'
next_day = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')

eth_df = yf.download("ETH-USD", start_date, today)
eth_df.tail()
eth_df.reset_index(inplace=True)
eth_df.columns

df = eth_df[["Date", "Open"]]

new_names = {
    "Date": "ds",
    "Open": "y",
}
df.rename(columns=new_names, inplace=True)
df.tail()
##plot open price##

x = df["ds"]
y = df["y"]

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y))
#set title
fig.update_layout(title_text="Time series plot of Ethereum Open Price")
fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all"),
                ]
            )
        ),
        rangeslider=dict(visible=True),
        type="date",
    )
)
# fig.show() 
# #shows graph

m = Prophet(seasonality_mode="multiplicative")
m.fit(df)
future = m.make_future_dataframe(periods=365)   #only dates
future.tail()                                   #only dates
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
eth_price_forecast = forecast[forecast['ds'] == next_day]['yhat'].item()
print (eth_price_forecast)

fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)

plt.show()
# plot_plotly(m, forecast)
# plot_components_plotly(m, forecast)
