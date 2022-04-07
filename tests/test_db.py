# Import Python libraries
import werkzeug
import logging
from flask import Flask, session, render_template, sessions, request, jsonify, Request
# from tensorflow.python.keras.utils.generic_utils import default
import numpy as np
from datetime import datetime as dt
import joblib
import os
from pyexpat import model
import pandas as pd
from sqlalchemy import create_engine, text, ForeignKey, DateTime, Table, Column, Integer, String
from sqlalchemy import select, update, delete, Column, Date, Integer, String
from sqlalchemy.orm import Session, relationship
from sqlalchemy.ext.declarative import declarative_base
from time import sleep
import numpy as np
import json
# Import own libraries
from data_api.db import return_engine, get_all_ticker_strings, get_ticker_by_ticker, delete_ticker, delete_model
from data_api.db import Model, get_ticker_by_id, create_model_db, get_model_by_id, create_ticker, create_price_data_set
from data_api.init_db import initialize_db, update_price_data_sets
from models.model_func import create_model, save_model, load_formatted_train_data, load_model, make_predictions

#initialize_db(start_date=dt(2022, 2, 10), end_date=dt.today())

# Define the data paths
current_dir = os.path.dirname(__file__)
database_dir = os.path.join(current_dir, "data")
db_filename = "database.db"
model_dir = os.path.join(current_dir, "data/models")
scaler_dir = os.path.join(current_dir, "data/scalers")

# Create the required folders, if they are not available
if not os.path.exists(database_dir):
    os.makedirs(database_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(scaler_dir):
    os.makedirs(scaler_dir)

# Get a connection to the database
engine = return_engine(database_dir, reset=True)

# Create some tickers for testing
aapl_id = create_ticker(engine, ticker="AAPL", name="Apple")
msft_id = create_ticker(engine, ticker="MSFT", name="Microsoft")
goog_id = create_ticker(engine, ticker="GOOG", name="Google")


aapl_1_id = create_price_data_set(engine, timestamp=dt.today(),
                                  open=10.6, close=11.5, high=13, low=9.5, adj_close=12, volume=15000, ticker_id=aapl_id)
aapl_2_id = create_price_data_set(engine, timestamp=dt.today(),
                                  open=10.6, close=11.5, high=13, low=9.5, adj_close=12, volume=68468, ticker_id=aapl_id)

msft_1_id = create_price_data_set(engine, timestamp=dt.today(),
                                  open=10.6, close=11.5, high=13, low=99.5, adj_close=12, volume=5898, ticker_id=msft_id)
msft_2_id = create_price_data_set(engine, timestamp=dt.today(),
                                  open=10.6, close=11.5, high=18, low=9.5, adj_close=12, volume=10000, ticker_id=msft_id)

goog_1_id = create_price_data_set(engine, timestamp=dt.today(),
                                  open=10.6, close=11.5, high=13, low=9.5, adj_close=12, volume=90, ticker_id=goog_id)
goog_2_id = create_price_data_set(engine, timestamp=dt.today(),
                                  open=10.6, close=11.5, high=13, low=9.5, adj_close=78, volume=15000, ticker_id=goog_id)


ticker_ids = [aapl_id, msft_id]
model_name = "AAPL_MSFT"
start_date = dt.today()
end_date = dt.today()
data_columns = pd.DataFrame(columns=["column1", "column2", "column3"]).columns
array = np.random.rand(1, 1, 60)


AAPL_MSFT_id = create_model_db(engine, model_name, start_date, end_date, tickers=ticker_ids, data_columns=data_columns,
                            last_data=array, scaler_path=scaler_dir + "test", model_path=model_dir + "test")

ticker_ids = [msft_id, goog_id]
model_name = "MSFT_GOOG"
start_date = dt.today()
end_date = dt(2020, 5, 1)
data_columns = pd.DataFrame(columns=["column1", "column2", "column3"]).columns
array = np.random.rand(1, 1, 60)

MSFT_GOOG_id = create_model_db(engine, model_name, start_date, end_date, tickers=ticker_ids, data_columns=data_columns,
                            last_data=array, scaler_path=scaler_dir + "test", model_path=model_dir + "test")

delete_ticker(engine, aapl_id)
delete_model(engine, AAPL_MSFT_id)
