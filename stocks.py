import datetime as date
import yfinance as yf
from prophet import Prophet


START = "2015-01-01"
TODAY = date.date.today().strftime("%Y-%m-%d")


def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

def fit_predict(data, period):

    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=period)
    forecast = model.predict(future)
    return forecast, model



