# Imports
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
# pickle
import pickle
import numpy as np

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

def setup_model_dict(ticker, name, start_date, end_date = datetime.today(), pred_base_range = 60):
    # Create dictionary to store the model and its metadata
    model_dict = {}
    model_dict["name"] = name
    model_dict["ticker"] = ticker
    model_dict["start_date"] = start_date
    model_dict["end_date"] = end_date
    model_dict["pred_base_range"] = pred_base_range
    
    # Get the data, rescale it and divide it into training and test set
    df = get_adj_close_df([ticker], start_date, end_date = datetime.today())
    df_values = df.values
    scaler, scaled_dataset = scale_data(df_values)
    x_train, y_train, x_test, y_test, training_data_len = train_test_split(scaled_dataset, days = pred_base_range)
    
    # Train a model with the data
    
    
    model = create_model(input_shape = x_train.shape[1])
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    model_dict["model"] = model
    
    return model_dict
    #store_model_keras("first_lstm", first_lstm)

# # Example call to create predictions
# # last_date_from_data, combined_dfs, original, df_forecast = get_prediction(tickers, target_var, start_date, end_date, n_days_for_prediction)

# class Model:
#     def __init__(self, _name, _engine, _ticker, _start_date, _end_date, _pred_base_range):
#         self.name = _name
#         self.engine = _engine
#         self.ticker = _ticker
#         self.start_date = _start_date
#         self.end_date = _end_date
#         self.pred_base_range = _pred_base_range





        
#     def save_on_disk(self, path = "./models/", filename = None):
#         # First store the model using the Keras function because the model cannot be stored with pickle (weakref-error)
#         store_model_keras(self.name + "_model", self.engine)
#         if filename == None:
#             fileObj = open(path + self.name + "_metadata.obj", 'wb')
#         else:
#             fileObj = open(path + filename + "_metadata.obj", 'wb')
#         pickle.dump(self, fileObj)
#         fileObj.close()
        
#     def load_model(self, filename, path = "./models/"):
#         with open(path + filename + "_metadata.obj", 'rb') as file:
#             # Call load method to deserialze
#             self = pickle.load(file) 
            
            
#     def predict(self, data):
#         predictions =  self.engine.predict(data).reshape(-1, 1)
#         return predictions
    
#     def predict_future(self, dataset,  n_days = 5, days_back = 60):
#         # Take the last n days from the dataset. This is the whole data needed for our prediction
#         last_n_days = dataset[-days_back:]
#         # Reshape so that the data can be used with our model
#         last_n_days = last_n_days.reshape(1, 60, 1)
#         # create new array which can hold the last 60 days and also the data to be predicted
#         last_n_days_array = np.zeros((1, days_back + n_days, 1))
#         # Copy the data to that new array
#         for i in range(last_n_days.shape[1]):
#             last_n_days_array[0][i][0] = last_n_days[0][i][0]
#         # Iterate through the new days which should be predicted and predict them based on the existing data and the previous predictions
#         for i in range(n_days):
#             # from 60 + 0 to 60 + 4:
#             last_60_temp = last_n_days_array[0, i : i  + days_back:, 0].reshape(1, 60 , 1)
#             # print(f"Last 60_temp: {last_60_temp}")
#             pred =  self.predict(last_60_temp)
#             #  print(f"Pred.shape: {pred.shape}")
#             #  print(f"Pred: {pred}")
#             last_n_days_array[0, days_back + i, 0]  = pred
#         return last_n_days_array[ - n_days: ,0]
    
#     def scaled_predict_future(self, dataset,  n_days = 5, days_back = 60):
#         # Take the last n days from the dataset. This is the whole data needed for our prediction
#         last_n_days = dataset[-days_back:]
#         # Rescale that data
#         scaler, scaled_data = scale_data(last_n_days)
#         # Reshape so that the data can be used with our model
#         scaled_data = scaled_data.reshape(1, 60, 1)
#         # create new array which can hold the last 60 days and also the data to be predicted
#         scaled_data_array = np.zeros((1, days_back + n_days, 1))
#         # Copy the data to that new array
#         for i in range(scaled_data.shape[1]):
#             scaled_data_array[0][i][0] = scaled_data[0][i][0]
#         # Iterate through the new days which should be predicted and predict them based on the existing data and the previous predictions
#         for i in range(n_days):
#             # from 60 + 0 to 60 + 4:
#             last_60_temp = scaled_data_array[0, i : i  + days_back:, 0].reshape(1, 60 , 1)
#             # print(f"Last 60_temp: {last_60_temp}")
#             pred =  self.predict(last_60_temp)
#             #  print(f"Pred.shape: {pred.shape}")
#             #  print(f"Pred: {pred}")
#             scaled_data_array[0, days_back + i, 0]  = pred

#         rescaled_prediction = rescale_data(scaled_data_array.reshape(-1, 1), scaler)
#         return rescaled_prediction[ - n_days: ,0]
    
#     def train(self, x_train, y_train, batch_size, epochs):
#         # model = create_model(input_shape = x_train.shape[1])
#         self.engine.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
        

# class LSTM_model(Model):
#     def __init__(self, _name, _ticker, _start_date, _end_date, _pred_base_range):
#         _engine = Sequential()
#         _engine.add(LSTM(units=50, return_sequences=True,input_shape=(_pred_base_range,1)))
#         _engine.add(LSTM(units=50, return_sequences=False))
#         _engine.add(Dense(units=25))
#         _engine.add(Dense(units=1))
#         _engine.compile(optimizer='adam', loss='mean_squared_error')
#         super(LSTM_model, self).__init__( _name, _engine, _ticker, _start_date, _end_date, _pred_base_range)

        
# # from sklearn.pipeline import Pipeline
# # from sklearn.preprocessing import MinMaxScaler

# # class LSTM_pipeline():
# #     def __init__(self, _model):
# #         return Pipeline(
# #             [
# #                 ('MinMaxScaler', MinMaxScaler()),
# #                 ('LSTM_Model', _model),
# #             ]
# #         )

    
