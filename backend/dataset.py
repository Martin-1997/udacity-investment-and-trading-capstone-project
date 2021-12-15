# Imports
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def split_dataset(dataset, days = 60, train_fraction = 0.7):
        # Get the lenght of 70% of thedata
        training_data_len = math.ceil(len(dataset) * train_fraction)
        # Get the training data
        train_data = dataset[0:training_data_len]

        # Separate the data into x and y data
        x_train = []
        y_train = []
        # Iterate from day 60 to the last day of the training data
        le = len(train_data)
        for i in range(days,len(train_data)):
            # I think these commands are useless
            # x_train=list(x_train)
            # y_train=list(y_train)
            x_train.append(train_data[i-days:i])
            y_train.append(train_data[i])

        # Converting the training x and y values to numpy arrays
        x_train, y_train = np.array(x_train), np.array(y_train)

        # Reshaping training s and y data to make the calculations easier
        # https://stats.stackexchange.com/questions/274478/understanding-input-shape-parameter-in-lstm-with-keras
        x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

        # Creating a dataset for testing
        test_data = dataset[training_data_len - days :]
        x_test = []
        y_test =  dataset[training_data_len : ]
        for i in range(days,len(test_data)):
            x_test.append(test_data[i-days:i])

        # 2.  Convert the values into arrays for easier computation
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

        return x_train, y_train, x_test, y_test

class Dataset:
    '''
    Class to store the data for a single ticker.
    '''
    def __init__(self, _ticker, _start_date, _end_date, _df, _all_days):
        '''
        Initializes a dataset instance.
        Parameters:
        - _ticker (string): The ticker symbol for the stock or asset used at Yahoo finance
        - _start_date (datetime.datetime): The first date for which data is included in the dataset
        - _end_date (datetime.datetime): The last date for which data is included in the dataset
        - _df (pandas.DataFrame): Pandas DataFrame which contains the data. The following columns must be included: [Open, Close, High, Low, Adj Close, Volume]
        - all_days (boolean): True if price data is available for all days (like for example for cryptocurrency data). False if there is not data for all days, f.e. not for weekends or holiday days
        '''
        self.ticker = _ticker
        self.start_date = _start_date
        self.end_date = _end_date
        self.df = _df
        self.all_days = _all_days
    
    def __str__(self):
        return f"Ticker: {self.ticker}; Start date: {self.start_date}; End date: {self.end_date}; All days: {self.all_days}; \n df: {self.df}"
    
    def plot_price_data(self):
        '''
        Plots the unadjusted prices for the "Open", "Close", "High" and "Low" prices.
        '''
        ax = self.df.loc[:,['Open','Close', 'High', 'Low']].plot(title = self.ticker, fontsize = 12)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price in $")
        plt.show()
        
    def plot_adj_close_data(self):
        '''
        Plots the adjusted close prices.
        '''
        ax = self.df.loc[:,['Adj Close']].plot(title = self.ticker, fontsize = 12)
        ax.set_xlabel("Date")
        ax.set_ylabel("Adjusted Price in $")
        plt.show()
        
    def plot_volume_data(self):
        '''
        Plots the volume data.
        '''
        ax = self.df.loc[:,['Volume']].plot(title = self.ticker, fontsize = 12)
        ax.set_xlabel("Date")
        ax.set_ylabel("Volume")
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.show()
        

    def train_test_split(self, column = "Adj Close", days = 60, train_fraction = 0.7):
        '''
        Returns a train and a test set base on the datasets data.
        
        PARAMETERS:
        
        - column (String - default: "Adj Close"): Which column data should be used
        
        - days (int - default: 60): Length of day range which is used to predict the next day
        
        - train_fraction (float, default: 0.7): Percentage of the data which should be used for training. The rest will be used for testing
        
        RETURNS:
        
        - x_train: An array of arrays with length "days". "days" consecutive days are stored in each array. Should be used for training a model.
        
        - y_train: An array of numbers specifing the true successor of each x_train array. Should be used for training a model.
        
        - x_test: An array of arrays with length "days". "days" consecutive days are stored in each array. Should be used for testing a model.
        
        - y_test: An array of numbers specifing the true successor of each x_train array. Should be used for testing a model.
        '''
        dataset = self.df[column].values
        dataset = dataset.reshape(-1, 1)
        x_train, y_train, x_test, y_test = split_dataset(dataset, days = days, train_fraction = train_fraction)
        return x_train, y_train, x_test, y_test
        
    
    def scaled_train_test_split(self, column = "Adj Close", days = 60, train_fraction = 0.7):
        dataset = self.df[column].values
        dataset = dataset.reshape(-1, 1)
        scaler = MinMaxScaler((0, 1))
        scaled_dataset = scaler.fit_transform(dataset)
        x_train, y_train, x_test, y_test = split_dataset(scaled_dataset, days = days, train_fraction = train_fraction)
        return x_train, y_train, x_test, y_test, scaler
    
    def return_row_array(self):
        return self.df.as_matrix()

def return_rmse(predictions, y_test): 
    '''
    Returns the Root Mean Square Error
    '''
    return np.sqrt(np.mean(((predictions- y_test)**2)))

def scale_data(data):
    '''
    Scales the data to a scale of 0 to 1
    
    PARAMETERS: 
    
    - data (array, list, pd.DataFrame ...): Data which should be scaled
    
    RETURNS:
    
    - scaler: The scaler which was used to scale the data. Is required to rescale the data or potential predictions based on the scaled data.
    - scaled_dataset: The scaled dataset
    '''
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Scale/Normalize the data to make all values between 0 and 1
    scaled_dataset = scaler.fit_transform(data)
    return scaler, scaled_dataset

def rescale_data(data, scaler):
    '''
    Rescales scaled data.
    
    PARAMETERS:
    
    - data (array, list, pd.DataFrame ...): Data which should be rescaled
    - scaler (Scaler): Scaler which was used to scale the data
    
    RETURNS:
    
    - rescaled_data: The rescaled data
    '''
    return scaler.inverse_transform(data)