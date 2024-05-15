import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import yfinance as yf

class Analyzer:
    def __init_(self):
        pass

    def download_data(self, stock_symbol, start_date, end_date) -> pd.DataFrame:
        """ Download stock data of given stock from Yahoo Finance API

        Returns Pandas dataframe that contains the selected stock's data between start_date and end_date

        :Parameters:
            stock_symbol : str
                Stock symbol of the stock to be downloaded
            start_date : datetime.date
                Start date of the stock date
            end_date : datetime.date
                End date of the stock data

        """
        self.start_date = start_date
        self.end_date = end_date

        try:
            # Download the data
            stock_data = yf.download(
                stock_symbol,
                start = start_date,
                end = end_date
            )
        except:
            return pd.DataFrame()

        stock_data = stock_data

        stock_data.index = pd.DatetimeIndex(stock_data.index).strftime("%Y-%m-%d")

        # Return the data, we don't need Adj Close since we have Close
        self.stock_data = stock_data.drop("Adj Close", axis = 1)
        return self.stock_data
    
    def show_close_price_in_time(self, stock_name):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x = self.stock_data.index, y = self.stock_data["Close"],name='Graph',mode='lines'))
        fig.update_layout(title = stock_name + "'s Close Price Between " + str(self.start_date) + " - " + str(self.end_date), xaxis_title='Date', yaxis_title="Close Price ($)")
        return fig
    
    def show_volume_close_in_time(self, stock_name):
        fig = make_subplots(specs=[[{"secondary_y": True}]])    
        fig.add_trace(go.Scatter(x = self.stock_data.index, y = self.stock_data["Close"],name='Close Price ($)',mode='lines'), secondary_y=False)
        fig.add_trace(go.Scatter(x = self.stock_data.index, y = self.stock_data["Volume"] / 100000, name='Volume (divided by 1 million)',mode='lines'), secondary_y=True)
        fig.update_layout(title = stock_name + "'s Close Price and Volume Between " + str(self.start_date) + " - " + str(self.end_date), xaxis_title='Date', yaxis_title="Close Price ($)")
        fig.update_yaxes(title_text="Volume", secondary_y=True)
        return fig