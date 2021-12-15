# Imports
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import access_api as api
from datetime import datetime, timedelta
#Libraries that will help us extract only business days in the US.
#Otherwise our dates would be wrong when we look back (or forward).  
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

# tickers = ["BTC-USD", "AAPL", "GOOG", "GC=F"]
# target_var = "AAPL"
# start_date = datetime(year = 2019, month = 11, day = 10)
# end_date = datetime(year = 2021, month = 11, day = 10)
# n_days_for_prediction=21

def get_prediction(tickers, target_var, start_date, end_date, n_days_for_prediction):
    df = api.get_adj_close_df(tickers, start_date, end_date)
    df = df.dropna() # Necessary when we have ticker symbols (f.e. Bitcoin) which has a price also for weekends - we need to drop these prices because stocks do not have prices on the weekend
    df["Date"] = df.index
    df.index = range(1, df.shape[0] + 1)

    #Separate dates for future plotting
    train_dates = pd.to_datetime(df['Date'])
    # print(train_dates.tail(15)) #Check last few dates. 

    #Variables for training
    cols = list(df)
    cols.remove("Date")

    #New dataframe with only training data - 5 columns
    df_for_training = df[cols].astype(float)

    print(f"Data columns used to build model: {cols}") #['Open', 'High', 'Low', 'Close', 'Adj Close']

    target_var_x_index = list(df_for_training).index(target_var)
    print(f"Column to predict: {cols[target_var_x_index]} (Index: {target_var_x_index})")
    print("\n")

    #LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
    # normalize the dataset
    scaler = StandardScaler()
    scaler = scaler.fit(df_for_training)
    df_for_training_scaled = scaler.transform(df_for_training)
    print(f"Dataset successfully scaled")
    print("\n")

    #As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features. 
    #In this example, the n_features is 5. We will make timesteps = 14 (past days data used for training). 

    #Empty lists to be populated using formatted training data
    trainX = []
    trainY = []

    n_future = 1   # Number of days we want to look into the future based on the past days.
    n_past = 60  # Number of past days we want to use to predict the future.

    #Reformat input data into a shape: (n_samples x timesteps x n_features)
    #In my example, my df_for_training_scaled has a shape (12823, 5)
    #12823 refers to the number of data points and 5 refers to the columns (multi-variables).
    for i in range(n_past, len(df_for_training_scaled) - n_future +1):
        # trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
        # trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])
        trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
        trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, target_var_x_index])


    trainX, trainY = np.array(trainX), np.array(trainY)

    print("Train dataset was successfully created:")
    print(f"\ttrainX shape == {trainX.shape}")
    print(f"\ttrainY shape == {trainY.shape}")
    print("\n")

    #In my case, trainX has a shape (12809, 14, 5). 
    #12809 because we are looking back 14 days (12823 - 14 = 12809). 
    #Remember that we cannot look back 14 days until we get to the 15th day. 
    #Also, trainY has a shape (12809, 1). Our model only predicts a single value, but 
    #it needs multiple variables (5 in my example) to make this prediction. 
    #This is why we can only predict a single day after our training, the day after where our data ends.
    #To predict more days in future, we need all the 5 variables which we do not have. 
    #We need to predict all variables if we want to do that. 

    # define the Autoencoder model
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
    model.add(LSTM(32, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(trainY.shape[1]))
    model.compile(optimizer='adam', loss='mse')
    print("Model was successfully created:")
    model.summary()
    print("\n")

    # fit the model
    history = model.fit(trainX, trainY, epochs=1, batch_size=16, validation_split=0.1, verbose=1)
    print("Model training successfull")
    print("\n")

    print("Model performance:")
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()
    print("\n")

    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())

    #Remember that we can only predict one day in future as our model needs 5 variables
    #as inputs for prediction. We only have all 5 variables until the last day in our dataset.
    n_past = 1 #TODO check what that variable does


    # list(train_dates)[-n_past + 1], <- the +1 to avoid to get a prediction for the last day in the timerange
    predict_period_dates = pd.date_range(list(train_dates)[-n_past - 1], periods=n_days_for_prediction, freq=us_bd).tolist()
    # print(predict_period_dates)

    #Make prediction
    prediction = model.predict(trainX[-n_days_for_prediction:]) #shape = (n, 1) where n is the n_days_for_prediction
    print("Prediction successfully created")
    print("\n")

    #Perform inverse transformation to rescale back to original range
    #Since we used 5 variables for transform, the inverse expects same dimensions
    #Therefore, let us copy our values 5 times and discard them after inverse transform
    prediction_copies = np.repeat(prediction, df_for_training.shape[1], axis=-1)
    y_pred_future = scaler.inverse_transform(prediction_copies)[:,target_var_x_index]

    # Convert timestamp to date
    forecast_dates = []
    for time_i in predict_period_dates:
        forecast_dates.append(time_i.date())
    df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), target_var :y_pred_future})
    df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])

    original = df[['Date', target_var]].copy()
    original['Date'] = pd.to_datetime(original['Date'])
    # original = original.loc[original['Date'] >= '2021-5-1']
    
    combined_dfs = original.append(df_forecast)
    last_date_from_data = original["Date"][original.index[-1]]

    # Create a forecast dataframe which contains the actual datapoint to avoid a gap when a graph is plotted
    df_forecast_new = original[original["Date"] == last_date_from_data]
    df_forecast_new = df_forecast_new.append(df_forecast)

    return last_date_from_data, combined_dfs, original, df_forecast_new


# last_date_from_data, combined_dfs, original, df_forecast = get_prediction(tickers, target_var, start_date, end_date, n_days_for_prediction)