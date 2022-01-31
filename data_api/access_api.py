from datetime import datetime as dt
from datetime import timedelta
# https://pydata.github.io/pandas-datareader/
import pandas_datareader as pdr
import pandas as pd


# # Stocks available to our app
# def get_tickers():
#     return [#"BTC-USD", # Bitcoin
#             "AAPL", # Apple
#             "GOOG", # Google
#             'GC=F', # Gold
#             'SI=F', # Silver
#             'EURUSD=X', # EUR-USD
#             'MSFT', # Microsoft
#             'AMD', # AMD
#             '^DJI', # Dow Jones Industrial average
#             '3333.HK', # China Evergrande Group
#             'ABNB', # AirBnB
#         ],["Bitcoin - USD",
#             "Apple",
#             "Google",
#             'Gold',
#             'Silver',
#             'EUR - USD',
#             'Microsoft',
#             'AMD',
#             'Dow Jones Industrial Average', 
#             'China Evergrande Group', 
#             'AirBnB'
#         ]

def get_nasdaq_tickers():
    return pdr.get_nasdaq_symbols()

def get_ticker_actions(ticker, start_date, end_date = dt.today):
    actions = pdr.data.DataReader(ticker, 'yahoo-actions', start_date, end_date)

def ticker_dividend(ticker, start_date, end_date = dt.today):
    dividends = pdr.data.DataReader(ticker, 'yahoo-dividends', start, end)
    return dividends

def get_stock_data(ticker, start_date, end_date = dt.today(), date_index = True, columns="all"):
    '''
    Returns stock data from yahoo for the specified ticker symbols, start and end dates.
    
    PARAMETERS:
    
    - ticker_symbols (list<string>): a list of standard ticker symbols which should be queried. Duplicate values will be ignored. If a ticker symbol could not be found, it is also going to be ignored and a error is printed.
    
    - start_date (dt.dt): the first date to fetch data for
    
    - end_date (dt.dt): the last date to fetch data for (default is the current date)
    
    - date_index (boolean): if true, the date-column will be used as the index for the dataframes (default). Otherwise, a numerical index is used and the date-date will be stored in a seperate column.
    
    - columns (list<string>): Determines which columns should be returned for each ticker (default = "all"). Options are "High", "Low", "Open", "Close" and "Volume"
    
    RETURNS:
    
    - data (dict<pandas.DataFrame>): A dictionary is returned and the data for each stock can be accessed by data["ticker_symbol"]
    '''
    try:
        df = pdr.data.DataReader(ticker, 'yahoo', start = start_date, end = end_date)
        if date_index:
            if "Date" in df.columns:
                df.set_index = df["Date"]
        else:
            df["Date"] = df.index
            df.index =  range(1, df.shape[0] + 1)
        if columns != "all":
            if len(columns) > 0:
                for column in df.columns:
                    if column not in columns:
                        df = df.drop([column], axis = 1)
        if df.isnull().sum().sum() != 0:
            print(f"WARN: {df.isnull().sum().sum() } data points are missing for ticker symbol {ticker}")
        return df
    except:
        print(f"Data could not be fetched for ticker {ticker}")


def get_adj_close_df(ticker_symbols, start_date, end_date = dt.today(), date_index = True, dropna_rows = True):
    """
    Returns a dataframe with the adjusted close prices for each ticker. Each column will represent the adjusted close price for single ticker.
    
    PARAMETERS:
    
    - ticker_symbols (list<string>): List of ticker symbols. Multiple occurances of ticker symbols will be ignored
    - start_date (dt.dt): Dateobject for the first adjusted price data point
    - end_date (dt.dt): Dateobject for the last adjusted price data point
    
    RETURNS:
    
    - df (dataframe): dataframe which contains the adjusted price data for all the ticker symbols
    """
    ticker_symbols = set(ticker_symbols)
    df = pd.DataFrame()
    for ticker in ticker_symbols:
        column = pdr.data.DataReader(ticker, 'yahoo', start = start_date, end = end_date)["Adj Close"]
        column.name = ticker
        df = pd.concat([df, column], axis = 1)

    if dropna_rows:
        # Necessary when we have ticker symbols (f.e. Bitcoin) which has a price also for weekends - we need to drop these prices because stocks do not have prices on the weekend
        df = df.dropna()

    if not date_index:
        # Extract the dates from the dataframe
        df["date"] = pd.to_dt(df.index)
        
        # Add a numerical index
        df.index = range(1, df.shape[0] + 1)
    return df


def tickers_as_columns(data_dict, column="Adj Close"):
        '''
        This method reformats a dictionary generated by the "get_stock_data"-method.
        The structure before is the following
        dict{
                stock1: [[High], [Low], [Close] ...],
                stock2: [[High], [Low], [Close] ...],
                ...
        }

        The method creates a dataframe which keeps the dates as the index, but adds a column for each stock based on a chosen value. The default value is "Adj Close".
        '''
        columns = []
        column_names = []
        for key in data_dict:
            column_names.append(key)
            columns.append(data_dict[key][column])
        df = pd.DataFrame(columns, index = column_names)
        return df.T


# def get_stock_data(ticker_symbols, start_date, end_date = dt.today(), date_index = True, columns="all"):
#     '''
#     Returns stock data from yahoo for the specified ticker symbols, start and end dates.
    
#     PARAMETERS:
    
#     - ticker_symbols (list<string>): a list of standard ticker symbols which should be queried. Duplicate values will be ignored. If a ticker symbol could not be found, it is also going to be ignored and a error is printed.
    
#     - start_date (dt.dt): the first date to fetch data for
    
#     - end_date (dt.dt): the last date to fetch data for (default is the current date)
    
#     - date_index (boolean): if true, the date-column will be used as the index for the dataframes (default). Otherwise, a numerical index is used and the date-date will be stored in a seperate column.
    
#     - columns (list<string>): Determines which columns should be returned for each ticker (default = "all"). Options are "High", "Low", "Open", "Close" and "Volume"
    
#     RETURNS:
    
#     - data (dict<pandas.DataFrame>): A dictionary is returned and the data for each stock can be accessed by data["ticker_symbol"]
#     '''
#     ticker_symbols = set(ticker_symbols)
#     data = {}
#     for ticker in ticker_symbols:
#         try:
#             df = pdr.data.DataReader(ticker, 'yahoo', start = start_date, end = end_date)
#             if date_index:
#                 if "Date" in df.columns:
#                     df.set_index = df["Date"]
#             else:
#                 df["Date"] = df.index
#                 df.index =  range(1, df.shape[0] + 1)
#             if columns != "all":
#                 if len(columns) > 0:
#                     for column in df.columns:
#                         if column not in columns:
#                             df = df.drop([column], axis = 1)
#             if df.isnull().sum().sum() != 0:
#                 print(f"WARN: {df.isnull().sum().sum() } data points are missing for ticker symbol {ticker}")
#             data[ticker] = df
#         except:
#             print(f"Data could not be fetched for ticker {ticker}")
#     return data

def get_stock_data_dict(ticker_symbols, start_date, end_date = dt.today(), date_index = True, columns="all"):
    '''
    Returns stock data from yahoo for the specified ticker symbols, start and end dates.
    A dictionary is returned and the data for each stock can be accessed by data["ticker_symbol"]
    '''
    ticker_symbols = set(ticker_symbols)
    data = {}
    for ticker in ticker_symbols:
        df = pdr.data.DataReader(ticker, 'yahoo', start = start_date, end = end_date)
        if date_index:
            if "Date" in df.columns:
                df.set_index = df["Date"]
        else:
            df["Date"] = df.index
            df.index =  range(1, df.shape[0] + 1)
        if columns != "all":
            if len(columns) > 0:
                for column in df.columns:
                    if column not in columns:
                        df = df.drop([column], axis = 1)
        if df.isnull().sum().sum() != 0:
            print(f"WARN: {df.isnull().sum().sum() } data points are missing for ticker symbol {ticker}")
        data[ticker] = df
    return data

def compute_daily_returns(df):
    '''
    Compute and return the daily return values
    '''
    daily_returns = df.copy()
    daily_returns[1:] = (df[1:] / df[:-1].values)-1
    daily_returns.iloc[0, :] = 0 # set daily returns of row 0 to 0
    return daily_returns