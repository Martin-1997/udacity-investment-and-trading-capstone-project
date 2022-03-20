from data_api.db import create_ticker, delete_ticker_by_ticker, get_ticker_by_id, return_engine, get_all_tickers, delete_all_price_data_sets, delete_price_data_set_by_ticker_id
from datetime import datetime as dt
from data_api.access_api import get_stock_data


def initialize_db(database_dir, db_filename="database.db", start_date=dt(2000, 1, 1), end_date=dt.today()):
    """
    Deletes the old database and sets up a new one with default data
    """
    engine = return_engine(database_dir, db_filename, reset=True)
    fill_initial_tickers(engine)
    update_price_data_sets(engine, start_date, end_date)
    return engine


def fill_initial_tickers(engine):
    """
    This method fills a set of predefined tickers into the specified database. This is used to make some data available initially.
    """
    create_ticker(engine, ticker="AAD.DE", name="Amadeus FiRe AG")
    create_ticker(engine, ticker="RWE.DE", name="RWE Aktiengesellschaft")
    create_ticker(engine, ticker="CVX", name="Chevron Corporation")
    create_ticker(engine, ticker="BTC-USD", name="Bitcoin")
    create_ticker(engine, ticker="AAPL", name="Apple")
    create_ticker(engine, ticker="GOOG", name="Google")
    create_ticker(engine, ticker="GC=F", name="Gold")
    create_ticker(engine, ticker="SI=F", name="Silver")
    create_ticker(engine, ticker="EURUSD=X'", name="EURO-USD")
    create_ticker(engine, ticker="MSFT", name="Microsoft")
    create_ticker(engine, ticker="EURUSD=X'", name="EURO-USD")
    create_ticker(engine, ticker="AMD", name="AMD")
    create_ticker(engine, ticker="^DJI", name="Dow Jones Industrial average")
    create_ticker(engine, ticker="3333.HK", name="China Evergrande Group")
    create_ticker(engine, ticker="ABNB", name="AirBnB")


def update_price_data_sets(engine, start_date=dt(2000, 1, 1), end_date=dt.today()):
    """
    Downloads the price data for the specified time range for all tickers and adds it to the database. Old data gets deleted.
    """
    # Get a list of all tickers
    all_tickers = get_all_tickers(engine)

    # Drop all data in the price_data_set table
    # https://stackoverflow.com/questions/16573802/flask-sqlalchemy-how-to-delete-all-rows-in-a-single-table

    # Deletes the old data in the price data sets table
    delete_all_price_data_sets(engine)

    # For each ticker
    for ticker in all_tickers:
        try:
            ticker_string = ticker[0].ticker
            ticker_id = ticker[0].id
            print(f"Data gets downloaded for ticker: {ticker_string}")

            # Load the available price dataset from the API
            data = get_stock_data(
                ticker_string, start_date, end_date, date_index=False)

            data["ticker_id"] = ticker_id

            data.rename(columns={
                        'High': 'high',
                        'Low': 'low',
                        'Open': 'open',
                        'Close': 'close',
                        'Volume': 'volume',
                        'Adj Close': 'adj_close',
                        'Date': 'timestamp',
                        }, inplace=True)

            # save the data to the database
            # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_sql.html
            print(
                f"Downloaded data for ticker {ticker_string} has the shape {data.shape}")
            data.to_sql(name="price_data_sets",
                        con=engine,
                        # Otherwise, rows with the same date (duplicates) will be appended. I am not sure how to solve this without havin
                        if_exists="append",
                        index=False,
                        index_label="id",
                        )
        except TypeError:
            print(
                f"Price data sets for ticker {ticker_string} with id {ticker_id} could not be downloaded.")
            print(f"Ticker {ticker_string} has been automatically deleted")
            delete_ticker_by_ticker(engine, ticker=ticker_string)


def update_ticker_price_data(engine, ticker_id, start_date=dt(2000, 1, 1), end_date=dt.today()):
    """
    Deletes all the existing price data for a ticker (if any) and downloads new data from the API to the database
    """
    ticker = get_ticker_by_id(engine, ticker_id)
    # Remove the existing price data
    delete_price_data_set_by_ticker_id(engine, ticker_id)
    ticker_string = ticker.ticker
    try:
        data = get_stock_data(
            ticker_string, start_date, end_date, date_index=False)

        data["ticker_id"] = ticker_id

        data.rename(columns={
                    'High': 'high',
                    'Low': 'low',
                    'Open': 'open',
                    'Close': 'close',
                    'Volume': 'volume',
                    'Adj Close': 'adj_close',
                    'Date': 'timestamp',
                    }, inplace=True)

        # save the data to the database
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_sql.html
        data.to_sql(name="price_data_sets",
                    con=engine,
                    # Otherwise, rows with the same date (duplicates) will be appended. I am not sure how to solve this without havin
                    if_exists="append",
                    index=False,
                    index_label="id",
                    )
    except:
        print(f"The data for ticker {ticker_string} could not be downloaded")
        print(f"Ticker {ticker_string} has been automatically deleted")
        delete_ticker_by_ticker(engine, ticker_string)
        return False
    return True
