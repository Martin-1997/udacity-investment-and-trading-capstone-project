from pyexpat import model
import pandas as pd
from sqlalchemy import BLOB, create_engine, text, ForeignKey, DateTime, Table, Column, Integer, String
from sqlalchemy import select, update, delete, Column, Date, Integer, String, PickleType
from sqlalchemy.orm import Session, relationship, joinedload
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime as dt
import json
import numpy as np
import os
import pandas_datareader as pdr


from data_api.access_api import get_stock_data


Base = declarative_base()


def return_engine(database_dir, db_filename="database.db", echo=False, reset=False):
    """
    Loads or creates a new database with the filename <db_filename>

    reset = True overwrites the existing data and creates a new database
    """
    db_path = os.path.join(database_dir, db_filename)
    engine = create_engine(
        f"sqlite+pysqlite:///{db_path}", echo=echo, future=True)
    if reset == True:
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    return engine


model_ticker = Table('model_ticker', Base.metadata,
                     Column('model_id', Integer, ForeignKey(
                         'models.id'), primary_key=True),
                     Column('ticker_id', Integer, ForeignKey(
                         'tickers.id'), primary_key=True)
                     )

# class ModelTicker(Base):
#     __tablename__ = 'model_ticker'
#     model_id = Column(Integer, ForeignKey('model.id'), primary_key=True),
#     ticker_id = Column(Integer, ForeignKey('ticker.id'), primary_key=True)


class Model(Base):
    """
    This class represents a machine learning model and its metadata
    """
    __tablename__ = "models"

    id = Column(Integer, primary_key=True)
    model_name = Column(String(30))
    start_date = Column(DateTime)
    end_date = Column(DateTime)

    data_columns = Column(String)
    last_data = Column(String)

    scaler_path = Column(String)
    model_path = Column(String)
    # If a model gets deleted, all the linked model_ticker entries get deleted -> no cascade rule required?
    tickers = relationship("Ticker",
                           secondary=model_ticker, back_populates="models")
    # ticker_ids = relationship("Ticker",
    #                           secondary=model_ticker, backref="models")
    # If a model gets deleted, all the linked predictions also get deleted
    # delete-orphan also deletes the child when it is deassociated from the parent
    prediction_ids = relationship(
        "Prediction", cascade="all, delete-orphan", backref="predictions")


class Ticker(Base):
    """
    This class represents a single asset
    """
    __tablename__ = "tickers"

    id = Column(Integer, primary_key=True)
    ticker = Column(String(10))
    name = Column(String(30))

    # If a ticker gets deleted, all the linked price_data_sets also get deleted
    # delete-orphan also deletes the child when it is deassociated from the parent
    price_data_sets = relationship(
        "Price_data_set", cascade="all, delete-orphan", backref="tickers")

    # If a ticker gets deleted, all the associated models need to be delted as well
    models = relationship("Model",
                          secondary=model_ticker, cascade="all, delete", back_populates="tickers")

    def __repr__(self):
        return f"Ticker(id={self.id!r}, ticker={self.ticker!r}, name={self.name!r})"


class Price_data_set(Base):
    """
    This class represents a dataset containing of a DateTime object and the corresponding price and volume data
    """
    __tablename__ = "price_data_sets"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime)
    open = Column(Integer)
    close = Column(Integer)
    high = Column(Integer)
    low = Column(Integer)
    adj_close = Column(Integer)
    volume = Column(Integer)

    ticker_id = Column(Integer, ForeignKey('tickers.id'))

    def __repr__(self):
        return f"Price Data Set(id={self.id!r}, timestamp={self.timestamp!r}, open={self.open!r}, close={self.close!r}, high={self.high!r}, low={self.low!r}, adj_close={self.adj_close!r}, volume={self.volume!r})"


class Prediction(Base):
    """
    This class represents a prediction made by a model
    """
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    tickers = Column(String)
    dates = Column(String)
    values = Column(String)

    model_id = Column(Integer, ForeignKey('models.id'))


# Models methods


def get_all_models(engine):
    """
    Returns a list of all model data stored in the model table
    """
    with Session(engine) as session:
        result = session.execute(select(Model)).all()
        for model in result:
            model[0].data_columns = json.loads(model[0].data_columns)
            model[0].last_data = np.array(json.loads(model[0].last_data))

    return result


def get_model_by_id(engine, model_id):
    """
    Returns a single model identified by its ID
    """
    with Session(engine) as session:
        # result = session.execute(select(Model).where(
        #     Model.id == model_id)).first()[0] # Otherwise, not the "real" model object is returned
        result = session.query(Model).options(joinedload(
            Model.tickers)).filter(Model.id == model_id).first()
        # If there is no model in the database, the result will be None
        if result == None:
            return None
        result.data_columns = json.loads(result.data_columns)
        result.last_data = np.array(json.loads(result.last_data))
    return result


def get_model_by_name(engine, model_name):
    """
    Returns a single model identified by its name
    """
    with Session(engine) as session:
        # session.expire_on_commit = False
        # result = session.execute(select(Model).where(
        #     Model.model_name == model_name)).first()[0] # Otherwise, not the "real" model object is returned
        # https://docs.sqlalchemy.org/en/14/orm/loading_relationships.html
        # https://docs.sqlalchemy.org/en/14/glossary.html#term-eager-loading
        # https://docs.sqlalchemy.org/en/14/orm/session_api.html#sqlalchemy.orm.Session.params.expire_on_commit

        result = session.query(Model).options(joinedload(Model.tickers)).filter(
            Model.model_name == model_name).first()
        result.data_columns = json.loads(result.data_columns)
        result.last_data = np.array(json.loads(result.last_data))
    return result


def create_model_db(engine, model_name, start_date, end_date, tickers, data_columns, last_data, scaler_path, model_path):
    """
    Add a new model to the database
    """
    if model_name_exists(engine, model_name):
        return -1

    with Session(engine) as session:
        # We need to query an instance for each ticker object from the database
        ticker_id_instances = get_ticker_instances(engine, tickers)
        data_columns = data_columns.tolist()
        # print(f"data_columns: {data_columns}, type: {type(data_columns)}")

        model = Model(model_name=model_name,
                      start_date=start_date, end_date=end_date,
                      tickers=ticker_id_instances,
                      data_columns=json.dumps(data_columns),
                      last_data=json.dumps(last_data.tolist()),
                      scaler_path=scaler_path,
                      model_path=model_path,
                      )
        session.add(model)
        session.commit()
        # Refresh the model to get its id
        session.refresh(model)
        id = model.id
        return id


def update_model(engine, model_id, model_name, start_date, end_date, ticker_ids, data_columns, last_data, scaler_path, model_path):
    """
    Update a model specified by its ID
    """
    with Session(engine) as session:
        # result = session.execute(select(Model).where(
        #     Model.id == model_id)).first()[0] # Otherwise, not the "real" model object is returned
        model = session.query(Model).options(joinedload(
            Model.tickers)).filter(Model.id == model_id).first()
        model.model_name = model_name
        model.start_date = start_date
        model.end_date = end_date
        model.tickers = ticker_ids
        model.data_columns = json.dumps(data_columns)
        model.last_data = json.dumps(last_data.tolist())
        model.scaler_path = scaler_path
        model.model_path = model_path
        session.commit()


def delete_model(engine, model_id):
    """
    Delete the model with the specified ID from the database
    """
    with Session(engine) as session:
        model = session.query(Model).filter(Model.id == model_id).first()
        session.delete(model)
        session.commit()


def delete_model_by_name(engine, model_name):
    """
    Delete the model with the specified name from the database
    """
    with Session(engine) as session:
        model = session.query(Model).filter(
            Model.model_name == model_name).first()
        session.delete(model)
        session.commit()


def delete_all_models(engine):
    with Session(engine) as session:
        # TODO check if cascade deletion works here
        models = session.query(Model).filter().all()
        print(models)
        for model in models:
            session.delete(model)
        session.commit()


def model_name_exists(engine, model_name):
    """
    Checks, if a model with a given name already exists
    """
    with Session(engine) as session:
        result = session.query(Model).options(joinedload(Model.tickers)).filter(
            Model.model_name == model_name).first()
        if result is None:
            return False
        else:
            return True


# Tickers methods


def get_all_tickers(engine):
    """
    Returns a list of all ticker data stored in the ticker table
    """
    with Session(engine) as session:
        result = session.execute(select(Ticker)).all()
    return result


def get_all_ticker_strings(engine):
    """
    Returns a list of all ticker strings stored in the ticker table
    """
    with Session(engine) as session:
        result = session.execute(select(Ticker)).all()

        return_items = []
        for item in result:
            return_items.append(item[0].ticker)
        return return_items


def get_ticker_by_ticker(engine, ticker):
    """
    Returns a single ticker identified by its ticker name
    """
    with Session(engine) as session:
        result = session.execute(select(Ticker).where(
            Ticker.ticker == ticker)).first()
    return result


def get_ticker_by_id(engine, ticker_id):
    """
    Returns a single ticker identified by its ID
    """
    with Session(engine) as session:
        # result = session.execute(select(Ticker).where(
        #     Ticker.id == ticker_id)).first()
        result = session.query(Ticker).filter(Ticker.id == ticker_id).first()
    return result


def get_ticker_instances(engine, ticker_ids):
    """
    This method returns the database instance for each ticker object with the specified ticker_id in ticker_ids
    """
    # We need to query an instance for each ticker object from the database
    with Session(engine) as session:
        ticker_id_instances = []
        for ticker_id in ticker_ids:
            ticker_id_instances.append(get_ticker_by_id(engine, ticker_id))
        return ticker_id_instances


def create_ticker(engine, ticker, name):
    with Session(engine) as session:
        ticker = Ticker(ticker=ticker, name=name)
        session.add(ticker)
        session.commit()
        # Refresh the ticker to get its id
        session.refresh(ticker)

        return ticker.id


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


def delete_ticker(engine, ticker_id):
    """
    Delete the ticker with the specified ID from the database
    """
    with Session(engine) as session:
        ticker = session.query(Ticker).filter(Ticker.id == ticker_id).first()
        session.delete(ticker)
        session.commit()


def delete_ticker_by_ticker(engine, ticker):
    """
    Delete the ticker with the specified ticker from the database
    """
    with Session(engine) as session:
        ticker = session.query(Ticker).filter(Ticker.ticker == ticker).first()
        session.delete(ticker)
        session.commit()


def delete_all_tickers(engine):
    with Session(engine) as session:
        session.query(Ticker).delete()
        session.commit()


def get_new_nasdaq_tickers(engine):
    """
    Returns a list of all NASDAQ tickers except those which already exist in the database
    """
    existing_ticker_strings = get_all_ticker_strings(engine)
    data = pdr.get_nasdaq_symbols()
    # Drop all rows which contain a ticker that is already in the database
    data = data[~data['NASDAQ Symbol'].isin(existing_ticker_strings)]
    tickers = data["NASDAQ Symbol"].tolist()
    asset_names = data["Security Name"].tolist()
    return tickers, asset_names


# Price data sets methods


def get_price_data_set(engine, id):
    with Session(engine) as session:
        # result = session.execute(select(Ticker).where(
        #     Ticker.id == ticker_id)).first()
        result = session.query(Price_data_set).filter(
            Price_data_set.id == id).first()
    return result


def get_all_price_data_sets(engine, ticker_ids=None, start_date=dt(1900, 1, 1), end_date=dt.today()):
    with engine.connect() as con:
        if ticker_ids == None:
            sql_alchemy_selectable = select(Price_data_set).where(start_date <= Price_data_set.timestamp).where(
                Price_data_set.timestamp <= end_date)
        else:
            sql_alchemy_selectable = select(Price_data_set).where(start_date <= Price_data_set.timestamp).where(
                Price_data_set.timestamp <= end_date).filter(Price_data_set.ticker_id.in_(ticker_ids))  # .where(Price_data_set.ticker_id in ticker_ids)

        data = pd.read_sql(sql=sql_alchemy_selectable,
                           con=con,
                           parse_dates=["timestamp"])
    return data


def get_formated_price_data_sets(engine, ticker_ids, start_date=dt(1900, 1, 1), end_date=dt.today()):
    """
    Returns a formated dataframe with the following columns:

    index | adj-close-{id1} | adj-close-id{2} | ... | volume-{id1} | volume-{id2} | ... | timestamp

    """
    # Drop all ticker_ids which are contained multiple times
    ticker_ids = set(ticker_ids)

    df = get_all_price_data_sets(engine, ticker_ids, start_date, end_date)
    # Drop the unnecessary columns
    df = df[["timestamp", "adj_close", "volume", "ticker_id"]]
    # Convert the ticker_id column to string - this is required to create new column names based on that column
    df["ticker_id"] = df["ticker_id"].apply(str)
    # Apply the pivot function, to create a multicolumn_index based on values and ticker_id, set timestamp as the index to avoid having this column twice
    # df = df.pivot(index="timestamp", columns="ticker_id", values=["adj_close", "volume"])
    pivot = pd.pivot_table(data=df, columns="ticker_id",
                           index="timestamp", values=['adj_close', 'volume'])
    # Create the new columns
    pivot.columns = ['-'.join(x) for x in pivot.columns]
    # Add timestamp as a column again
    pivot["timestamp"] = pivot.index
    # Add a numerical index
    pivot.index = range(1, pivot.shape[0] + 1)
    return pivot


def create_price_data_set(engine, timestamp, open, close, high, low, adj_close, volume, ticker_id):
    if open < 0:
        raise ValueError('The open price needs to be non-negative')
    if close < 0:
        raise ValueError('The close price needs to be non-negative')
    if high < 0:
        raise ValueError('The high price needs to be non-negative')
    if low < 0:
        raise ValueError('The low price needs to be non-negative')
    if adj_close < 0:
        raise ValueError('The adj_close price needs to be non-negative')
    if volume < 0:
        raise ValueError('The volume needs to be non-negative')
        
    with Session(engine) as session:
        price_data_set = Price_data_set(
            timestamp=timestamp,
            open=open,
            close=close,
            high=high,
            low=low,
            adj_close=adj_close,
            volume=volume,
            ticker_id=ticker_id,
        )
        session.add(price_data_set)
        session.commit()
        # Refresh the price_data_set to get its id
        session.refresh(price_data_set)

        return price_data_set.id


def update_price_data_set(engine, id, timestamp, open, close, high, low, adj_close, volume, ticker_id):
    if open < 0:
        raise ValueError('The open price needs to be non-negative')
    if close < 0:
        raise ValueError('The close price needs to be non-negative')
    if high < 0:
        raise ValueError('The high price needs to be non-negative')
    if low < 0:
        raise ValueError('The low price needs to be non-negative')
    if adj_close < 0:
        raise ValueError('The adj_close price needs to be non-negative')
    if volume < 0:
        raise ValueError('The volume needs to be non-negative')

    with Session(engine) as session:
        data = session.execute(
            select(Price_data_set).filter_by(id=id)).scalar_one()
        data.timestamp = timestamp
        data.open = open
        data.close = close
        data.high = high
        data.low = low
        data.adj_close = adj_close
        data.volume = volume
        data.ticker_id = ticker_id
        session.commit()

        return id


def delete_price_data_set(engine, id):
    """
    Delete the Price_data_set with the specified ID from the database
    """
    with Session(engine) as session:
        price_data_set = session.query(Price_data_set).filter(
            Price_data_set.id == id).first()
        id = price_data_set.id
        session.delete(price_data_set)
        session.commit()
        return id


def delete_all_price_data_sets(engine):
    with Session(engine) as session:
        session.query(Price_data_set).delete()
        session.commit()


def delete_price_data_set_by_ticker_id(engine, ticker_id):
    with Session(engine) as session:
        price_data_sets = session.query(Price_data_set).filter(
            Price_data_set.ticker_id == ticker_id).first()
        if price_data_sets is not None:
            session.delete(price_data_sets)
            session.commit()
        return ticker_id


def load_formatted_train_data(engine, ticker_ids, start_date, end_date):
    """
    Loads the required raw data from the database
    """
    # Get the data from the API
    # df = api.get_adj_close_df(tickers, start_date, end_date, date_index=False)

    # Get the data from the database
    df = get_formated_price_data_sets(engine, ticker_ids, start_date, end_date)

    # Extract the dates from the dataframe
    dates = df["timestamp"]
    df.drop(["timestamp"], axis=1, inplace=True)

    # New dataframe with only training data
    df_for_training = df.astype(float)
    print(f"Data columns used to build model: {df.columns.values}")
    return df_for_training


# Prediction methods


def get_all_predictions(engine):
    with Session(engine) as session:
        result = session.execute(select(Prediction)).all()


def get_prediction_by_id(engine, id):
    with Session(engine) as session:
        result = session.query(Prediction).filter(Prediction.id == id).first()
        return result


def get_predictions_by_model_id(engine, model_id):
    with Session(engine) as session:
        result = session.query(Prediction).filter(
            Prediction.model_id == model_id).all()
        return result


def get_predictions_by_model_name(engine, model_name):
    with Session(engine) as session:
        result = session.query(Prediction).filter(
            Prediction.model_name == model_name).all()
        return result


def create_prediction(engine, tickers, dates, values, model_id):
    with Session(engine) as session:
        prediction = Prediction(
            tickers=json.dumps(tickers.tolist()),
            dates=json.dumps(dates.tolist()),
            values=json.dumps(values),
            model_id=model_id,
        )
        session.add(prediction)
        session.commit()
        # Refresh the prediction to get its id
        session.refresh(prediction)

        return prediction.id


def update_prediction(engine, id, tickers, dates, values, model_id):
    with Session(engine) as session:
        # result = session.execute(select(Model).where(
        #     Model.id == model_id)).first()[0] # Otherwise, not the "real" model object is returned
        prediction = session.query(Prediction).filter(
            Prediction.id == id).first()
        prediction.tickers = json.dumps(tickers.tolist())
        prediction.dates = json.dumps(dates.tolist())
        prediction.values = json.dumps(values)
        prediction.model_id = model_id
        session.commit()


def delete_prediction(engine, id):
    with Session(engine) as session:
        prediction = session.query(Prediction).filter(
            Prediction.id == id).first()
        session.delete(prediction)
        session.commit()
        return id


def delete_predictions_by_model(engine, model_id):
    with Session(engine) as session:
        predictions = session.query(Prediction).filter(
            Prediction.model_id == model_id).first()
        session.delete(predictions)
        session.commit()
        return model_id


def delete_all_predictions(engine):
    with Session(engine) as session:
        session.query(Prediction).delete()
        session.commit()
        return True
