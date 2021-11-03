# Imports
from datetime import datetime, timedelta
# https://pydata.github.io/pandas-datareader/
import pandas_datareader as pdr
import pandas as pd


def get_stock_data(ticker, start_date, end_date = datetime.today(), date_index = True, columns="all"):
    '''
    Returns stock data from yahoo for the specified ticker symbols, start and end dates.
    
    PARAMETERS:
    
    - ticker_symbols (list<string>): a list of standard ticker symbols which should be queried. Duplicate values will be ignored. If a ticker symbol could not be found, it is also going to be ignored and a error is printed.
    
    - start_date (datetime.datetime): the first date to fetch data for
    
    - end_date (datetime.datetime): the last date to fetch data for (default is the current date)
    
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


def get_adj_close_df(ticker_symbols, start_date, end_date = datetime.today()):
    """
    Returns a dataframe with the adjusted close prices for each ticker. Each column will represent the adjusted close price for single ticker.
    
    PARAMETERS:
    
    - ticker_symbols (list<string>): List of ticker symbols. Multiple occurances of ticker symbols will be ignored
    - start_date (datetime.datetime): Dateobject for the first adjusted price data point
    - end_date (datetime.datetime): Dateobject for the last adjusted price data point
    
    RETURNS:
    
    - df (dataframe): dataframe which contains the adjusted price data for all the ticker symbols
    """
    ticker_symbols = set(ticker_symbols)
    df = pd.DataFrame()
    for ticker in ticker_symbols:
        column = pdr.data.DataReader(ticker, 'yahoo', start = start_date, end = end_date)["Adj Close"]
        column.name = ticker
        df = pd.concat([df, column], axis = 1)
   # df = pd.DataFrame(data = columns) #, columns=ticker_symbols, index = range(1, len(columns[1])))
    return df


# def get_stock_data(ticker_symbols, start_date, end_date = datetime.today(), date_index = True, columns="all"):
#     '''
#     Returns stock data from yahoo for the specified ticker symbols, start and end dates.
    
#     PARAMETERS:
    
#     - ticker_symbols (list<string>): a list of standard ticker symbols which should be queried. Duplicate values will be ignored. If a ticker symbol could not be found, it is also going to be ignored and a error is printed.
    
#     - start_date (datetime.datetime): the first date to fetch data for
    
#     - end_date (datetime.datetime): the last date to fetch data for (default is the current date)
    
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
