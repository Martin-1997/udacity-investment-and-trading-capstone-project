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
from datetime import date


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
    # If non-negative return values are required, this should be accomplished by the layers in the network. Anyway, with only non-negative input values, negative output values are very unlikely.
    # The relu activation function only returns positive values. -> This returns the same values for each predicted date
    model.add(Dense(output_shape[2]))  # ,  W_constraint=nonneg()))
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
    # Reformat input data into a shape: (n_samples x timesteps x n_features)
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

    assert(data.shape[0] > n_past),  f"Training not possible. A minimum of {n_past} days is needed to train the model. Only {data.shape[0]} days have been seleted."

    # Normalize the dataset
    scaler = StandardScaler()
    scaler = scaler.fit(data)
    df_for_training_scaled = scaler.transform(data)
    print(f"Dataset successfully scaled. n_features of the scaler: {scaler.n_features_in_}")

    trainX, trainY = create_train_test_arrays(
        n_past=n_past, df=df_for_training_scaled)

    model = get_model(input_shape=trainX.shape,
                      output_shape=trainY.shape, print_summary=False)

    # fit the model
    history = model.fit(trainX, trainY, epochs=1,
                        batch_size=16, validation_split=0.1, verbose=1)
    print("Model training successfull")

    return model, trainX[-1], scaler, data.columns


def make_predictions(model, last_date_model, last_data, scaler, n_days, data_columns):
    """
    Creates and returns predictions for the next n_days days
    """
    data = []

    # Within this step, an additional dimension is added
    three_dim_last_data = last_data.reshape(
        1, last_data.shape[0], last_data.shape[1])

    i = 1
    for i in range(0, n_days): # len(forecast_dates)):
        # Make prediction
        prediction = model.predict(three_dim_last_data)
        # Perform inverse transformation to rescale back to original range
        scaled_prediction = scaler.inverse_transform(prediction)
        # We set the date later as an index
        # scaled_prediction = np.append(scaled_prediction, forecast_dates[i])
        # Insert the prediction into the dataframe
        #df.loc[i] = scaled_prediction
        data.append(scaled_prediction)
        # Insert the prediction as last row into our dataframe
        prediction = prediction.reshape(
            1, prediction.shape[0], prediction.shape[1])
        three_dim_last_data = np.append(
            three_dim_last_data, prediction, axis=1)
        # Drop the first row in the dataframe to again have the same amount of rows
        three_dim_last_data = three_dim_last_data[:,
                                                  1: three_dim_last_data.shape[1] + 1]
    # len(forecast_dates)
    data = np.reshape(data, [n_days, len(data_columns)])
    df = pd.DataFrame(data=data,  columns=data_columns) # index=pd.to_datetime(forecast_dates),
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
    

def convert_to_business_days(date_list):
    """
    Converts all non-business days to the next business day. Drops duplicates which exist already or which appear during the conversion.
    """
    date_list = pd.to_datetime(date_list)
    # Expand the time range by 1 into the past and into the future to avoid problems with the range creation, if only a single date is selected
    # In the case of a single date, who is not a business date, this date will be anyway returned as the only member of the range
    business_days = pd.bdate_range(
        min(date_list) - pd.DateOffset(1), max(date_list) + pd.DateOffset(1))
    new_date_list = []
    for date_obj in date_list:
        if date_obj in business_days:
            # print(f"{date_obj} is a business day! \n")
            new_date_list.append(date_obj)
        else:
            next_business_day = date_obj + pd.offsets.BDay()
            # print(f"{date_obj} is not a business day! The next business day is {next_business_day} \n")
            new_date_list.append(next_business_day)
    return list(set(new_date_list))
