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

class Model:
    def __init__(self, _name, _engine, _ticker, _start_date, _end_date, _pred_base_range):
        self.name = _name
        self.engine = _engine
        self.ticker = _ticker
        self.start_date = _start_date
        self.end_date = _end_date
        self.pred_base_range = _pred_base_range
        
    def save_on_disk(self, path = "./models/", filename = None):
        # First store the model using the Keras function because the model cannot be stored with pickle (weakref-error)
        store_model_keras(self.name + "_model", self.engine)
        if filename == None:
            fileObj = open(path + self.name + "_metadata.obj", 'wb')
        else:
            fileObj = open(path + filename + "_metadata.obj", 'wb')
        pickle.dump(self, fileObj)
        fileObj.close()
        
    def load_model(self, filename, path = "./models/"):
        with open(path + filename + "_metadata.obj", 'rb') as file:
            # Call load method to deserialze
            self = pickle.load(file) 
            
            
    def predict(self, data):
        predictions =  self.engine.predict(data).reshape(-1, 1)
        return predictions
    
    def predict_future(self, dataset,  n_days = 5, days_back = 60):
        # Take the last n days from the dataset. This is the whole data needed for our prediction
        last_n_days = dataset[-days_back:]
        # Reshape so that the data can be used with our model
        last_n_days = last_n_days.reshape(1, 60, 1)
        # create new array which can hold the last 60 days and also the data to be predicted
        last_n_days_array = np.zeros((1, days_back + n_days, 1))
        # Copy the data to that new array
        for i in range(last_n_days.shape[1]):
            last_n_days_array[0][i][0] = last_n_days[0][i][0]
        # Iterate through the new days which should be predicted and predict them based on the existing data and the previous predictions
        for i in range(n_days):
            # from 60 + 0 to 60 + 4:
            last_60_temp = last_n_days_array[0, i : i  + days_back:, 0].reshape(1, 60 , 1)
            # print(f"Last 60_temp: {last_60_temp}")
            pred =  self.predict(last_60_temp)
            #  print(f"Pred.shape: {pred.shape}")
            #  print(f"Pred: {pred}")
            last_n_days_array[0, days_back + i, 0]  = pred
        return last_n_days_array[ - n_days: ,0]
    
    def scaled_predict_future(self, dataset,  n_days = 5, days_back = 60):
        # Take the last n days from the dataset. This is the whole data needed for our prediction
        last_n_days = dataset[-days_back:]
        # Rescale that data
        scaler, scaled_data = scale_data(last_n_days)
        # Reshape so that the data can be used with our model
        scaled_data = scaled_data.reshape(1, 60, 1)
        # create new array which can hold the last 60 days and also the data to be predicted
        scaled_data_array = np.zeros((1, days_back + n_days, 1))
        # Copy the data to that new array
        for i in range(scaled_data.shape[1]):
            scaled_data_array[0][i][0] = scaled_data[0][i][0]
        # Iterate through the new days which should be predicted and predict them based on the existing data and the previous predictions
        for i in range(n_days):
            # from 60 + 0 to 60 + 4:
            last_60_temp = scaled_data_array[0, i : i  + days_back:, 0].reshape(1, 60 , 1)
            # print(f"Last 60_temp: {last_60_temp}")
            pred =  self.predict(last_60_temp)
            #  print(f"Pred.shape: {pred.shape}")
            #  print(f"Pred: {pred}")
            scaled_data_array[0, days_back + i, 0]  = pred

        rescaled_prediction = rescale_data(scaled_data_array.reshape(-1, 1), scaler)
        return rescaled_prediction[ - n_days: ,0]
    
    def train(self, x_train, y_train, batch_size, epochs):
        # model = create_model(input_shape = x_train.shape[1])
        self.engine.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
        

class LSTM_model(Model):
    def __init__(self, _name, _ticker, _start_date, _end_date, _pred_base_range):
        _engine = Sequential()
        _engine.add(LSTM(units=50, return_sequences=True,input_shape=(_pred_base_range,1)))
        _engine.add(LSTM(units=50, return_sequences=False))
        _engine.add(Dense(units=25))
        _engine.add(Dense(units=1))
        _engine.compile(optimizer='adam', loss='mean_squared_error')
        super(LSTM_model, self).__init__( _name, _engine, _ticker, _start_date, _end_date, _pred_base_range)

        
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import MinMaxScaler

# class LSTM_pipeline():
#     def __init__(self, _model):
#         return Pipeline(
#             [
#                 ('MinMaxScaler', MinMaxScaler()),
#                 ('LSTM_Model', _model),
#             ]
#         )

    
