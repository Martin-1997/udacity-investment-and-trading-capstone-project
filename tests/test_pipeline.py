# Import Python libraries
import werkzeug
import logging
from flask import Flask, session, render_template, sessions, request, jsonify, Request
# from tensorflow.python.keras.utils.generic_utils import default
import numpy as np
from datetime import datetime as dt
from datetime import timedelta
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

# Allow importing of own libraries from the parent directory
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# Import own libraries
from data_api.db import return_engine, get_all_ticker_strings, get_ticker_by_ticker, delete_ticker, delete_model
from data_api.db import Model, get_ticker_by_id, create_model_db, get_model_by_id, create_ticker, create_price_data_set, load_formatted_train_data
from data_api.init_db import initialize_db, update_price_data_sets
from models.model_func import create_model, save_model, load_model, make_predictions


# Define the data paths
current_dir = os.path.dirname(__file__)
database_dir = os.path.join(current_dir, "../data")
db_filename = "database.db"
model_dir = os.path.join(current_dir, "../data/models")
scaler_dir = os.path.join(current_dir, "../data/scalers")

# Create the required folders, if they are not available
if not os.path.exists(database_dir):
    os.makedirs(database_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(scaler_dir):
    os.makedirs(scaler_dir)

if os.path.exists(os.path.join(database_dir, db_filename)):
    # If the database already exits, get a connection to the database
    engine = return_engine(database_dir, db_filename=db_filename)
else:
    # Otherwise, create a new database and initialize it with data
    engine = initialize_db(database_dir,  db_filename=db_filename)

msft_id = get_ticker_by_ticker(engine, "MSFT")[0].id
goog_id = get_ticker_by_ticker(engine, "GOOG")[0].id

ticker_ids = [goog_id, msft_id]
model_name = "GOOG_MSFT"
start_date = dt(2021, 2, 2)
end_date = dt.today()

# Convert the tickers into ticker_ids
model_tickers_ids = []
for ticker in ticker_ids:
    model_tickers_ids.append(
        get_ticker_by_id(engine, ticker).id)

print(f"start date: {start_date}; end date: {end_date}")
print(f"model_tickers_ids: {model_tickers_ids}")

train_data = load_formatted_train_data(
                engine, model_tickers_ids, start_date, end_date)


train_data.fillna(value=0, inplace=True)

print("")
print("")
print(f"Train data shape: {train_data.shape}")
print("")
print("")

model, last_data, data_columns = create_model(
                train_data)





# AAPL_MSFT_id = create_model_db(engine, model_name, start_date, end_date, tickers=ticker_ids, data_columns=data_columns,
#                             last_data=array, scaler_path=scaler_dir + "test", model_path=model_dir + "test")

# ticker_ids = [msft_id, goog_id]
# model_name = "MSFT_GOOG"
# start_date = dt.today()
# end_date = dt.today() - timedelta(days = 180)
# data_columns = pd.DataFrame(columns=["column1", "column2", "column3"]).columns
# array = np.random.rand(1, 1, 60)

# MSFT_GOOG_id = create_model_db(engine, model_name, start_date, end_date, tickers=ticker_ids, data_columns=data_columns,
#                             last_data=array, scaler_path=scaler_dir + "test", model_path=model_dir + "test")

# delete_ticker(engine, aapl_id)
# delete_model(engine, AAPL_MSFT_id)
