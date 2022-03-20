# Imports
import numpy as np
import matplotlib as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import StandardScaler
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar


def print_performance(history):
    print("Model performance:")
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()
    print("\n")


def get_model(input_shape, output_shape, print_summary=True):
    """
    Returns a predefined model object
    """
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(
        input_shape[1], input_shape[2]), return_sequences=True))
    model.add(LSTM(32, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(output_shape[2]))
    model.compile(optimizer='adam', loss='mse')
    print("Model was successfully created:")
    if print_summary:
        print(model.summary())
    return model


def create_train_test_arrays(n_past, df):
    """
    Takes the data and creates trainX and trainY datasets. n_past values are used to predict the value at index n_past + 1
    """
    # Prepare training and testing data
    # Empty lists to be populated using formatted training data
    trainX = []
    trainY = []
    # #Reformat input data into a shape: (n_samples x timesteps x n_features)
    for i in range(n_past, len(df)):
        trainX.append(df[i - n_past:i, 0:df.shape[1]])
        trainY.append(df[i:i + 1])
    trainX, trainY = np.array(trainX), np.array(trainY)
    print("Train dataset was successfully created:")
    print(f"trainX shape == {trainX.shape}")
    print(f"trainY shape == {trainY.shape}")
    return trainX, trainY


def create_model(data,  n_past=60):
    """
    Prepares the data and creates and trains a model
    """

    assert(
        data.shape[0] > n_past),  f"Training not possible. A minimum of {n_past} days is needed to train the model. Only {data.shape[0]} days have been seleted."

    # LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
    # normalize the dataset
    scaler = StandardScaler()
    scaler = scaler.fit(data)
    df_for_training_scaled = scaler.transform(data)
    print(
        f"Dataset successfully scaled. n_features of the scaler: {scaler.n_features_in_}")

    # As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features.
    # In this example, the n_features is 5. We will make timesteps = 14 (past days data used for training).
    trainX, trainY = create_train_test_arrays(
        n_past=n_past, df=df_for_training_scaled)

    model = get_model(input_shape=trainX.shape,
                      output_shape=trainY.shape, print_summary=False)

    # fit the model
    history = model.fit(trainX, trainY, epochs=1,
                        batch_size=16, validation_split=0.1, verbose=1)
    print("Model training successfull")

    # print_performance(history)

    return model, trainX[-1], scaler, data.columns


def create_prediction_date_range(last_date_model, n_days):
    """
    Outputs the dates for the next n_days business days
    """
    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    dates_to_predict = pd.date_range(
        last_date_model, periods=n_days, freq=us_bd).tolist()
    # Convert timestamp to date
    forecast_dates = []
    for time_i in dates_to_predict:
        forecast_dates.append(time_i.date())
    return forecast_dates


def make_predictions(model, last_date_model, last_data, scaler, n_days, data_columns):
    """
    Creates and returns predictions for the next n_days days
    """
    # Get the range of dates to predict values for
    forecast_dates = create_prediction_date_range(
        last_date_model, n_days)

    # Add a data column
    data_columns.append("date")

    # create an empty result dataframe to store all the predictions later
    df = pd.DataFrame(columns=data_columns)

    # Within this step, an additional dimension is added
    three_dim_last_data = last_data.reshape(
        1, last_data.shape[0], last_data.shape[1])

    i = 1
    for i in range(0, len(forecast_dates)):
        # Make prediction
        prediction = model.predict(three_dim_last_data)
        # Perform inverse transformation to rescale back to original range
        scaled_prediction = scaler.inverse_transform(prediction)
        scaled_prediction = np.append(scaled_prediction, forecast_dates[i])
        # Insert the prediction into the dataframe
        df.loc[i] = scaled_prediction
        # Insert the prediction as last row into our dataframe
        prediction = prediction.reshape(
            1, prediction.shape[0], prediction.shape[1])
        three_dim_last_data = np.append(
            three_dim_last_data, prediction, axis=1)
        # Drop the first row in the dataframe to again have the same amount of rows
        three_dim_last_data = three_dim_last_data[:,
                                                  1: three_dim_last_data.shape[1] + 1]
    return df


def save_model(model, filename, path="./data/models/", extension=".h5"):
    """
    Stores the model to disk and returns the full path including the name and extension
    """
    path = f"{path}{filename}{extension}"
    model.save(path)
    return path


def load_model(name, path="./data/models/", extension=".h5"):
    """
    Loads the model from disk
    """
    return keras.models.load_model(f'{path}{name}{extension}')
