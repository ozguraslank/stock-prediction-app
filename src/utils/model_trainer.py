import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from catboost import CatBoostRegressor
import numpy as np

def ARMA_ARIMA_model(train_data: pd.DataFrame, pred_data: pd.DataFrame, adf_test_score: int) -> pd.DataFrame:
    """ Implement ARMA or ARIMA model to the stock data
           
        Returns predictions for next 90 business days

        :Parameters:
            train_data : Pandas dataframe
                Stock data that will be used as train data
            pred_data : Pandas dataframe    
                Stock data that will be used as prediction data, this contains both test and future historical data
            adf_test_score : int
                ADF test score of the stock data, If the score is 1, differencing method will be applied on data, If Its 0, wont be applied
    """
    train_data_df = train_data.copy()
    pred_df = pred_data.copy()


    # auto_arima function helps us to find the best parameters for ARIMA, since we already use adf_test_score as d value, we can tell function to ignore it
    # Also, due to the operation of auto_arima function, It should be called just before the operation, that's why its being imported here
    from pmdarima.arima import auto_arima
    auto_arima = auto_arima(train_data_df["Close"], max_p = 2, max_q = 2, d = None, stepwise=False, seasonal=False)

    # After the test, the parameters are being located in the auto_arima object
    best_params = auto_arima.get_params()["order"]
    best_p = best_params[0]  # p value
    best_q = best_params[2]  # q value
    
    # Training the ARMA, ARIMA model
    model = ARIMA(train_data_df["Close"], order=(best_p, adf_test_score, best_q))
    model = model.fit()
    
    # Make future predictions for next 90 business days
    start = len(train_data_df)
    end = start + len(pred_df) - 1

    predictions = model.predict(start = start, end = end)

    # Create a new dataframe that contains the predictions
    predictions_df = pd.DataFrame({"Close": predictions})
    predictions_df.reset_index(drop=True, inplace=True)

    pred_df["Close"] = predictions_df.values

    # Return the predictions
    return pred_df
    

def prophet_model(train_data: pd.DataFrame, pred_data: pd.DataFrame, adf_test_score: int) -> pd.DataFrame:
    """ Implement Prophet model to the stock data
           
        Returns predictions for next 90 business days

        :Parameters:
            train_data : Pandas dataframe
                Stock data that will be used as train data
            pred_data : Pandas dataframe
                Stock data that contains the dates that prediction will be made
            adf_test_score : int
                ADF test score of the stock data
    """

    train_data_df = train_data.copy()

    # Since the Prophet model requires only Date and y (Close price) as columns, we are filtering the columns
    train_data_df = train_data_df[["Date", "Close"]]
    pred_df = pred_data.copy()

    # Rename the columns to the default column names for Prophet model, Prophet model requires Date as ds, Close as y
    train_data_df.rename(columns = {"Date":"ds", "Close":"y"}, inplace = True)
    pred_df.rename(columns = {"Date":"ds", "Close":"y"}, inplace = True)

    # If the data is stationary, differencing is not needed
    differenced_check = 0

    # If the data is not stationary, condition block will be executed and differencing will be implemented on the data
    if (adf_test_score):
        last_value = train_data_df.iloc[-1]["y"]
        #train_data_differenced = train_data.copy()
        train_data_df["y"] = train_data_df["y"].diff().dropna()
        differenced_check = 1   # check

    # Model creation and fit the model
    prophet = Prophet(growth = "linear")
    prophet.fit(train_data_df)

    # Make future predictions for next 90 days
    
    # Data processing on predictions; Removing unneccesary columns, leaving only the predictions that created for test_data and after (Prophet model makes the prediction for whole train_data, that's why) 
    predictions = prophet.predict(pred_df)  
    predictions = predictions[['ds', 'yhat']]

    # If differencing is implemented on the data, we should reconvert the predictions to normal values  
    if (differenced_check):
        predictions["yhat"] = predictions["yhat"].cumsum()
        predictions["yhat"] = predictions["yhat"].add(last_value)

    # Rename the columns to the default column names for all predictions
    predictions.rename(columns = {"ds":"Date", "yhat":"Close"}, inplace = True)

    # Return the predictions 
    return predictions


def catboost_model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, pred_data: pd.DataFrame) -> pd.DataFrame:
    """ Implement CatBoost model to the stock data
           
        Returns predictions for next 90 business days

        :Parameters:
            train_data : Pandas dataframe
                Stock data that will be used as train data
            
            pred_data : Pandas dataframe
                Stock data that contains the dates that prediction will be made
            adf_test_score : int
                ADF test score of the stock data
    
    """
    catboost = CatBoostRegressor(iterations = 500, eval_metric = "RMSE", allow_writing_files = False)
    catboost.fit(X_train, y_train["Close"], early_stopping_rounds = 250, verbose = 50)
    pred = catboost.predict(pred_data)
    
    pred_df = pd.DataFrame(pred, columns = ["Close"])

    return pred_df
