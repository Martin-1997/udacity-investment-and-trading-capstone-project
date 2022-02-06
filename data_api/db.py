import pandas as pd
from sqlalchemy import create_engine, text, ForeignKey, DateTime, Table, Column, Integer, String
from sqlalchemy import select, update, delete, Column, Date, Integer, String
from sqlalchemy.orm import Session, relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime as dt

from data_api.access_api import get_stock_data


Base = declarative_base()


def initialize_db(start_date=dt(2000, 1, 1), end_date=dt.today()):
    """
    Deletes the old database and sets up a new one with default data
    """
    engine = return_engine(reset=True)
    fill_initial_tickers(engine)
    update_price_data_sets(engine, start_date, end_date)


def return_engine(db_filename="database.db", echo=False, reset=False):
    """
    Loads or creates a new database with the filename <db_filename>

    reset = True overwrites the existing data and creates a new database
    """
    engine = create_engine(
        f"sqlite+pysqlite:///data/{db_filename}", echo=echo, future=True)
    if reset == True:
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    return engine


class Ticker(Base):
    """
    This class represents a single asset
    """
    __tablename__ = "ticker"

    id = Column(Integer, primary_key=True)
    ticker = Column(String(10))
    name = Column(String(30))

    price_data_sets = relationship("Price_data_set", back_populates="ticker")

    def __repr__(self):
        return f"Ticker(id={self.id!r}, ticker={self.ticker!r}, name={self.name!r})"


class Price_data_set(Base):
    """
    This class represents a dataset containing of a DateTime object and the corresponding price and volume data
    """
    __tablename__ = "price_data_set"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime)
    open = Column(Integer)
    close = Column(Integer)
    high = Column(Integer)
    low = Column(Integer)
    adj_close = Column(Integer)
    volume = Column(Integer)

    ticker_id = Column(Integer, ForeignKey('ticker.id'))

    ticker = relationship("Ticker", back_populates="price_data_sets")

    def __repr__(self):
        return f"Price Data Set(id={self.id!r}, timestamp={self.timestamp!r}, open={self.open!r}, close={self.close!r}, high={self.high!r}, low={self.low!r}, adj_close={self.adj_close!r}, volume={self.volume!r})"


def fill_initial_tickers(engine):
    """
    This method fills a set of predefined tickers into the specified database. This is used to make some data available initially.
    """
    with Session(engine) as session:
        amadeus = Ticker(ticker="AAD.DE", name="Amadeus FiRe AG")
        session.add(amadeus)

        rwe = Ticker(ticker="RWE.DE", name="RWE Aktiengesellschaft")
        session.add(rwe)

        chevron = Ticker(ticker="CVX", name="Chevron Corporation")
        session.add(chevron)

        btc_usd = Ticker(ticker="BTC-USD", name="Bitcoin")
        session.add(btc_usd)

        aapl = Ticker(ticker="AAPL", name="Apple")
        session.add(aapl)

        goog = Ticker(ticker="GOOG", name="Google")
        session.add(goog)

        gold = Ticker(ticker="GC=F", name="Gold")
        session.add(gold)

        silver = Ticker(ticker="SI=F", name="Silver")
        session.add(silver)

        euro_usd = Ticker(ticker="EURUSD=X'", name="EURO-USD")
        session.add(euro_usd)

        microsoft = Ticker(ticker="MSFT", name="Microsoft")
        session.add(microsoft)

        amd = Ticker(ticker="AMD", name="AMD")
        session.add(amd)

        dow_jones = Ticker(ticker="^DJI", name="Dow Jones Industrial average")
        session.add(dow_jones)

        evergrande = Ticker(ticker="3333.HK", name="China Evergrande Group")
        session.add(evergrande)

        airbnb = Ticker(ticker="ABNB", name="AirBnB")
        session.add(airbnb)

        session.commit()


def update_price_data_sets(engine, start_date=dt(2000, 1, 1), end_date=dt.today(), delete_old=False):
    """
    Downloads the price data for the specified time range for all tickers and adds it to the database.

    delete_old: If True, all the old price data in the database is going to be deleted before inserting the new one
    """
    # Get a list of all tickers
    all_tickers = get_ticker_objects(engine)

    # Drop all data in the price_data_set table
    # https://stackoverflow.com/questions/16573802/flask-sqlalchemy-how-to-delete-all-rows-in-a-single-table

    with Session(engine) as session:
        if delete_old:
            session.query(Price_data_set).delete()
        session.commit()

        # For each ticker
        for ticker in all_tickers:
            try:
                ticker_string = ticker[0].ticker
                ticker_id = ticker[0].id
                print(f"Ticker: {ticker_string}")
                print(f"Ticker_id: {ticker_id}")

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

                print(f"Columns: {data.columns}")
                # save the data to the database
                # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_sql.html
                data.to_sql(name="price_data_set",
                            con=engine,
                            if_exists="append",
                            index=False,
                            )
                session.commit()
            except TypeError:
                print(
                    f"Price data sets for ticker {ticker_string} with id {ticker_id} could not be downloaded.")

##############################################
# QUERIES
##############################################

# Tickers


def get_ticker_objects(engine):
    """
    Returns a list of all ticker data stored in the ticker table
    """
    with Session(engine) as session:
        result = session.execute(select(Ticker)).all()
    return result


def get_ticker_strings(engine):
    """
    Returns a list of all ticker strings stored in the ticker table
    """
    with Session(engine) as session:
        result = session.execute(select(Ticker.ticker)).all()

    return_items = []
    for item in result:
       # print(f"{item[0]} \n")
        return_items.append(item[0])
    return return_items


def get_ticker_by_id(engine, ticker_id):
    """
    Returns a single ticker identified by its ID
    """
    with Session(engine) as session:
        result = session.execute(select(Ticker).where(
            Ticker.id == ticker_id)).first()

    return result


def load_ticker_by_ticker(engine, ticker):
    """
    Returns a single ticker identified by its ticker name
    """
    with Session(engine) as session:
        result = session.execute(select(Ticker).where(
            Ticker.ticker == ticker)).first()
    return result


def delete_ticker_by_id(engine, ticker_id):
    """
    Delete the ticker with the specified ID from the database
    """
    with Session(engine) as session:
        session.execute(delete(Ticker).where(Ticker.id == ticker_id))
        session.commit()


def delete_ticker_by_ticker(engine, ticker):
    """
    Delete the ticker with the specified ticker name from the database
    """
    with Session(engine) as session:
        session.execute(delete(Ticker).where(Ticker.ticker == ticker))
        session.commit()

# Price data sets


def get_price_data_sets(engine, ticker_ids=None, start_date=dt(1900, 1, 1), end_date=dt.today()):
    with engine.connect() as con:
        if ticker_ids == None:
            sql_alchemy_selectable = select(Price_data_set).where(
                start_date <= Price_data_set.timestamp <= end_date)
        else:
            sql_alchemy_selectable = select(Price_data_set).where(start_date <= Price_data_set.timestamp).where(
                Price_data_set.timestamp <= end_date).filter(Price_data_set.ticker_id.in_(ticker_ids))  # .where(Price_data_set.ticker_id in ticker_ids)

        data = pd.read_sql(sql=sql_alchemy_selectable,
                           con=con,
                           parse_dates=["timestamp"])
    return data


def get_formated_data(engine, ticker_ids, start_date=dt(1900, 1, 1), end_date=dt.today()):
    """
    Returns a formated dataframe with the following columns:

    index | adj-close-{id1} | adj-close-id{2} | ... | volume-{id1} | volume-{id2} | ... | timestamp

    """
    df = get_price_data_sets(engine, ticker_ids, start_date, end_date)
    # Drop the unnecessary columns
    df = df[["timestamp", "adj_close", "volume", "ticker_id"]]
    # Convert the ticker_id column to string - this is required to create new column names based on that column
    df["ticker_id"] = df["ticker_id"].apply(str)
    # Apply the pivot function, to create a multicolumn_index based on values and ticker_id, set timestamp as the index to avoid having this column twice
    df = df.pivot(index="timestamp", columns="ticker_id",
                  values=["adj_close", "volume"])
    # Create the new columns
    df.columns = ['-'.join(x) for x in df.columns]
    # Add timestamp as a column again
    df["timestamp"] = df.index
    # Add a numerical index
    df.index = range(1, df.shape[0] + 1)
    return df


def update_ticker(engine, ticker_id, ticker, name):
    """
    Updates the ticker with the specified id using the provided ticker and name
    """
    with Session(engine) as session:
        data = session.execute(
            select(Ticker).filter_by(id=ticker_id)).scalar_one()
        data.ticker = ticker
        data.name = name
        session.commit()
