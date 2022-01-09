# Imports
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import keras
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import backend.access_api as api
from datetime import datetime, timedelta
#Libraries that will help us extract only business days in the US.
#Otherwise our dates would be wrong when we look back (or forward).  
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

# tickers = ["BTC-USD", "AAPL", "GOOG", "GC=F"]
# target_var = "AAPL"
# start_date = datetime(year = 2019, month = 11, day = 10)
# end_date = datetime(year = 2021, month = 11, day = 10)
# n_days_for_prediction = 21


def print_performance(history):
    print("Model performance:")
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()
    print("\n")


def get_model(input_shape, output_shape, print_summary = True):
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(input_shape[1], input_shape[2]), return_sequences=True))
    model.add(LSTM(32, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(output_shape[1]))
    model.compile(optimizer='adam', loss='mse')
    print("Model was successfully created:")
    if print_summary:
        print(model.summary())
    return model


def create_train_test_arrays(n_past, df):
    # Prepare training and testing data
    #Empty lists to be populated using formatted training data
    trainX = []
    trainY = []
    # #Reformat input data into a shape: (n_samples x timesteps x n_features)
    # print(f"Iteration to create training arrays:")
    # print(f"for i in range(n_past, len(df_for_training_scaled) - n_future +1)")
    # print(f"for i in range({n_past}, {len(df_for_training_scaled)} - {n_future} +1)")
    for i in range(n_past, len(df)):
        trainX.append(df[i - n_past:i, 0:df.shape[1]])
        trainY.append(df[i:i + 1])
    trainX, trainY = np.array(trainX), np.array(trainY)
    print("Train dataset was successfully created:")
    print(f"trainX shape == {trainX.shape}")
    print(f"trainY shape == {trainY.shape}")
    return trainX, trainY


def create_model(tickers, start_date, end_date,  n_past = 60):
    # Get the data from the API
    df = api.get_adj_close_df(tickers, start_date, end_date, date_index=False)
    
    # Extract the dates from the dataframe
    dates = df["date"]
    df.drop(["date"], axis=1, inplace=True)

    # New dataframe with only training data
    df_for_training = df.astype(float)
    print(f"Data columns used to build model: {df.columns.values}")

    #LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
    # normalize the dataset
    scaler = StandardScaler()
    scaler = scaler.fit(df_for_training)
    df_for_training_scaled = scaler.transform(df_for_training)
    # print(f"Dataset successfully scaled. n_features of the scaler: {scaler.n_features_in_}")

    #As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features. 
    #In this example, the n_features is 5. We will make timesteps = 14 (past days data used for training). 
    trainX, trainY = create_train_test_arrays(n_past=n_past, df = df_for_training_scaled)

    print(f"model input shape: {trainX.shape}")
    print(f"model output shape: {trainY.shape}")
    model = get_model(input_shape = trainX.shape, output_shape = trainY.shape, print_summary = False)

    # fit the model
    history = model.fit(trainX, trainY, epochs=1, batch_size=16, validation_split=0.1, verbose=1)
    print("Model training successfull")

    # print_performance(history)

    return model, trainX[-1], scaler


def create_modelname(model_tickers, startdate, enddate):
    # Create a model name
    modelname = ""
    for i  in range(len(model_tickers)):
        if i < len(model_tickers) - 1:
            modelname += model_tickers[i] + "_"
        else:
            modelname += model_tickers[i] + "-"
    modelname += str(startdate) + "-"
    modelname += str(enddate)
    return modelname


def make_predictions(model, last_date_model, last_data, scaler, n_days_for_prediction, model_tickers):
    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())

    #Remember that we can only predict one day in future as our model needs 5 variables
    #as inputs for prediction. We only have all 5 variables until the last day in our dataset.
    # n_past = 1 #TODO check what that variable does


    # list(dates)[-n_past + 1], <- the +1 to avoid to get a prediction for the last day in the timerange
    # dates_to_predict = pd.date_range(list(dates)[-1 - 1], periods=n_days_for_prediction, freq=us_bd).tolist()
    dates_to_predict = pd.date_range(last_date_model, periods=n_days_for_prediction, freq=us_bd).tolist()
    # Convert timestamp to date
    forecast_dates = []
    for time_i in dates_to_predict:
        forecast_dates.append(time_i.date())

    # get price data of the last n_past days
    # for date in dates_to_predict:
    print(f"Dates for which we want to predict a value: {forecast_dates}")

    #Make prediction
    # prediction = model.predict(trainX[-n_days_for_prediction:]) #shape = (n, 1) where n is the n_days_for_prediction
    prediction = model.predict(last_data)
    print("Prediction successfully created")

    #Perform inverse transformation to rescale back to original range
    #Since we used 5 variables for transform, the inverse expects same dimensions
    #Therefore, let us copy our values 5 times and discard them after inverse transform
    # prediction_copies = np.repeat(prediction, df_for_training.shape[1], axis=-1)
    # prediction_copies = np.repeat(prediction, scaler.n_features_in_, axis=-1)
    # y_pred_future = scaler.inverse_transform(prediction_copies)[:,target_var_x_index]

    y_pred_future = scaler.inverse_transform(prediction ) # [:] #,target_var_x_index]
    print(f"Shape of y_pred_future: {y_pred_future.shape}")

    
    df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), model_tickers :y_pred_future})
    df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])

    # original = df[['Date', target_var]].copy()
    # original['Date'] = pd.to_datetime(original['Date'])
    # original = original.loc[original['Date'] >= '2021-5-1']
    
    # combined_dfs = original.append(df_forecast)
    # last_date_from_data = original["Date"][original.index[-1]]

    # # Create a forecast dataframe which contains the actual datapoint to avoid a gap when a graph is plotted
    # df_forecast_new = original[original["Date"] == last_date_from_data]
    # df_forecast_new = df_forecast_new.append(df_forecast)

    return df_forecast


def save_model(model, filename):
    print("Save method")
    path = f"./models/{filename}.h5" #{filename}"
    print(path)
    model.save(path)
    # model.save(f'./models/{filename}')


def load_model(name):
    return keras.models.load_model(f'./models/{name}.h5')


# Example call to create predictions
# last_date_from_data, combined_dfs, original, df_forecast = get_prediction(tickers, target_var, start_date, end_date, n_days_for_prediction)