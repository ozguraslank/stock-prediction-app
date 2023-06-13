import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import datetime as dt
from Models import models

from sklearn.metrics import mean_squared_error


from statsmodels.tsa.stattools import adfuller  # ADF Test

import warnings
warnings.filterwarnings('ignore')

class Model:
    def __init__(self):
        pass


    def main(self, stock_symbol: str):
        self.stock_symbol = stock_symbol
        self.stock_data = self.download_data()                     # Download data
        self.adf_test_score = self.adf_test()                      # Do ADF test
        self.train_data, self.test_data = self.train_test_split(self.stock_data)  # Split the data into train and test
        self.pred_data = self.create_pred_data(self.test_data)     # Create future data for the prediction
        self.pred_df = self.fit_models(self.train_data, self.pred_data)   # Fit models; ARMA-ARIMA, Prophet, and LSTM
        self.best_model_df = self.compare_results()                # Compare the prediction results of the models and select the best one according to the RMSE score

        # Since prediction data also has the test_data in it too, we should delete it and leave only the future dates in it
        last_date = self.stock_data["Date"].tail(1).iloc[0]                                 # Deletion will be till last element in actual data
        self.best_model_df = self.best_model_df.loc[self.best_model_df["Date"] > last_date] # Delete till last actual data's date

        #self.best_model_df.index += len(self.train_data)
        column_names = list(self.best_model_df.columns)
        self.best_model_df.rename(columns = {column_names[1]: "Close"}, inplace = True)
        self.best_model_df.set_index("Date", inplace = True)
        self.stock_data.set_index("Date", inplace = True)

        #self.final_df = pd.concat([self.stock_data, self.best_model_df])
        #self.final_df.set_index("Date", inplace = True)
        #self.final_df.index = pd.DatetimeIndex(self.final_df.index).strftime("%Y-%m-%d")

        #print(self.best_model_df)
        return self.best_model_df
    

    def download_data(self) -> pd.DataFrame:
        """ Download stock data of given stock from Yahoo Finance API

        Returns Pandas dataframe that contains the selected stock's data between 01-01-2020 and today 

        :Parameters:
            stock_symbol : str
                Stock symbol of the stock to be downloaded

        """

        start_date = dt.date(2020, 1, 1)
        end_date = dt.date.today()

        # Download the data
        stock_data = yf.download(
            self.stock_symbol,
            start = start_date,
            end = end_date
        )

        stock_data = stock_data[["Close"]]

        stock_data.reset_index(inplace = True)

        # Return the data
        return stock_data
    

    def adf_test(self) -> int:
        """ Implement adf test to the stock data, If the data is stationary (p_value < 0.05) It will return 0, If not, will return 1
           
        the return value will be used for ARIMA(p, d (return), q) in order to decide fitting whether ARMA or ARIMA model

        :Parameters:
            stock_data : Pandas dataframe
                Stock data to be tested
        """

        adf_result = adfuller(self.stock_data["Close"])
        p_value = adf_result[1]  #  If p value is less than .05, then data is stationary, if not, it's not.

        return 0 if p_value < 0.05 else 1


    def train_test_split(self, stock_data: pd.DataFrame) -> list:
        """ Split the stock data into train and test

        Test data will contain the last 15 days of stock data
        
        Returns train and test data

        :Parameters:
            stock_data : Pandas dataframe
                Stock data that will be splitted into train and test
        
        """
        test_size = 15
        

        # Split the data into train and test data
        train_data = stock_data[:len(stock_data) - test_size]
        test_data = stock_data[len(stock_data) - test_size:]

        return [train_data, test_data]
    

    def create_pred_data(self, test_data) -> pd.DataFrame:
        """ Create and return a dataframe that contains the dates of the test data and approximately next 90 business days from it

        *Business days* are the days that stock market is open

        
        :Parameters:
            test_data : Pandas dataframe
                Test data of the stock data
        
        """
        
        # Create date dataframe of next 120 days, since the stock market is not open in Weekend and national holidays, 120 days are equal to approximately 90 business days
        pred_df = pd.DataFrame(pd.date_range(start = test_data["Date"].head(1).iloc[0], periods = 150, freq = "D"), columns = ["Date"])
        #pred_df.index += test_data.index[0]

        pred_df = self.remove_holidays_weekends(pred_df)

        return pred_df
    
    def remove_holidays_weekends(self, prediction_df: pd.DataFrame) -> pd.DataFrame:
        """ Remove the predictions that made for holidays and weekends, since the stock market is closed on those days, they should be deleted.

        :Parameters:
            prediction_df : Pandas dataframe
                prediction data for a stock for t time
        """

        prediction_df = prediction_df[prediction_df['Date'].dt.dayofweek < 5]   # Removing the weekend days from the predictions

        # List of US Federal Holidays in 2023
        holidays = [
        '2023-01-01',  # New Year's Day
        '2023-01-16',  # Martin Luther King, Jr. Day
        '2023-02-20',  # Presidents Day (Washington's Birthday)
        '2023-04-07',  # Good Friday
        '2023-05-29',  # Memorial Day
        '2023-06-19',  # Juneteenth National Independence Day
        '2023-07-04',  # Independence Day
        '2023-09-04',  # Labor Day
        '2023-11-23',  # Thanksgiving Day
        '2023-12-25'   # Christmas Day
        ]   

        # Convert the list of holidays to a pandas DataFrame
        holidays_df = pd.DataFrame({'Date': pd.to_datetime(holidays)})

        # Convert the 'date' column to pandas datetime format if it's not already
        prediction_df['Date'] = pd.to_datetime(prediction_df['Date'])

        # Filter out the holidays from the prediction dataframe
        filtered_prediction_df = prediction_df[~prediction_df['Date'].isin(holidays_df['Date'])]
        filtered_prediction_df['Date'] = pd.to_datetime(filtered_prediction_df['Date'])

        filtered_prediction_df = filtered_prediction_df.reset_index(drop=True)
        
        return filtered_prediction_df
    
    def catboost_data_preparation(self, train_data: pd.DataFrame, pred_data: pd.DataFrame):
        future_days = 150
        new_df = self.stock_data
        extra_day = pd.DateOffset(days=1)
        pred_date = pd.DataFrame(pd.date_range(start = new_df["Date"].tail(1).iloc[0] + extra_day, periods = future_days, freq = "D"), columns = ["Date"])
        pred_date = self.remove_holidays_weekends(pred_date)
        pred_date.index += len(new_df)
        new_df = pd.concat([new_df, pred_date])
    
        # Defining the parameters for rolling and lags
        rolling_windows = [future_days, future_days+5, future_days+14, future_days+20, future_days+25, future_days+30]
        lags = [future_days, future_days+5, future_days+14, future_days+20, future_days+25, future_days+30] 

        for i in rolling_windows:
            new_df["rolling_mean_" + str(i)] = new_df["Close"].rolling(i, min_periods = 1).mean().shift(1)
            new_df["rolling_std_" + str(i)] = new_df["Close"].rolling(i, min_periods = 1).std().shift(1)
            new_df["rolling_min_" + str(i)] = new_df["Close"].rolling(i, min_periods = 1).min().shift(1)
            new_df["rolling_max_" + str(i)] = new_df["Close"].rolling(i, min_periods = 1).max().shift(1)
            new_df["rolling_var_" + str(i)] = new_df["Close"].rolling(i, min_periods = 1).var().shift(1)

        df_lag = new_df.copy()

        for l in lags:
            new_df["Close_lag_" + str(l)] = df_lag["Close"].shift(l)

        new_df.dropna(subset=new_df.columns.difference(["Close"]), inplace=True)
  
        test_size = 15

        train_end = pred_data["Date"].head(1).iloc[0]
        historical_data = new_df.loc[new_df["Date"] < train_end]
        X = historical_data.drop("Close", axis = 1).set_index("Date")
        y = historical_data[["Date", "Close"]].set_index("Date")

        forecast_df = new_df.loc[new_df["Date"] > train_end].set_index("Date").drop("Close", axis = 1)
        X_train,  X_test = X[:len(X) - test_size], X[(len(X) - test_size): len(X)]
        y_train,  y_test = y[:len(y) - test_size], y[(len(y) - test_size): len(y)]
        
        return [X_train, X_test, y_train, forecast_df]


    def fit_models(self, train_data: pd.DataFrame, pred_data: pd.DataFrame):
        # Fitting ARMA or ARIMA models
        ARMA_ARIMA_pred = models.ARMA_ARIMA_model(
            train_data,
            pred_data,
            self.adf_test_score,
        )

        
        # Fitting Prophet model
        prophet_pred = models.prophet_model(
            train_data,
            pred_data,
            self.adf_test_score
        )

        catboost_params = self.catboost_data_preparation(train_data, pred_data)

        catboost_pred = models.catboost_model(
            catboost_params[0],
            catboost_params[1],
            catboost_params[2],
            catboost_params[3],
        )

        catboost_pred = catboost_pred[:len(prophet_pred)]

        pred_df = ARMA_ARIMA_pred.copy().rename(columns = {"Close": "ARMA-ARIMA"})
        pred_df["Prophet"] = prophet_pred["Close"].values
        pred_df["CatBoost"] = catboost_pred["Close"].values

        pred_df.set_index("Date", inplace = True)
        
        return pred_df
    
    
    def calculate_RMSE(self, test_data: pd.DataFrame, preds: pd.DataFrame) -> float:
        """ Calculate the RMSE score of the predictions

        Returns RMSE score of the predictions

        :Parameters:
            test_data : Pandas dataframe
                Test data of the stock
            preds : Pandas dataframe
                Predictions of the stock
        """


        # Calculate the RMSE score
        RMSE = round(np.sqrt(mean_squared_error(test_data["Close"], preds[:len(test_data)])), 2)

        return RMSE
    
    
    def compare_results(self) -> pd.DataFrame:
        """ Compare the results of the models

        Returns the best model's predictions as Pandas Dataframe

        """

        # The prediction dataframe that contains all the models predictions
        pred_df = self.pred_df.copy()


        # RMSE scores of each model will be added in this dictionary, the format will be, MODEL_NAME: RMSE_SCORE
        RMSE_scores = {}

        for column_name, column_data in pred_df.iteritems():
            RMSE_scores[column_name] = self.calculate_RMSE(self.test_data, column_data)

        # Converting dictionary into dataframe
        RMSE_scores_df = pd.DataFrame(RMSE_scores.values(), columns = ["RMSE Score"], index = RMSE_scores.keys())

        # Finding the best model according to the lowest RMSE, we need the column name which is the model name, so we will retrieve it via the pred_df
        best_model = RMSE_scores_df.idxmin().values[0]

        # Creating a dataframe that contains the best model's predictions   
        best_model_df = pred_df[[best_model]]
        
        best_model_df.reset_index(inplace = True)

        return best_model_df
    

    def show_actual_pred(self, stock_name):
        """ Show the actual and predicted values of the stock in a line graph

        :Parameters:
            stock_name : string
                Name of the stock
        """
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x = self.stock_data.index, y = self.stock_data["Close"],name='Actual',mode='lines'))
        fig.add_trace(go.Scatter(x = self.best_model_df.index, y = self.best_model_df["Close"],name='Prediction',mode='lines'))
        fig.update_layout(title = "Future prediction for " + stock_name, xaxis_title='Date', yaxis_title="Close Price ($)")
        return fig
    
    def prepare_final_df(self):
        pass
            

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



def main():
    model = Model()
    
    model.main("^OEX")



if __name__ == "__main__":
    main()





