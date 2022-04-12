# Import Python libraries
from unicodedata import name
import werkzeug
from flask import Flask, session, render_template, sessions, request, jsonify, Request, url_for, redirect
import numpy as np
from datetime import date, datetime
import joblib
import os
from flask_socketio import SocketIO, emit
# For accessing sqlalchemy exceptions
import sqlalchemy
import pandas as pd

# Import own libraries
from data_api.db import delete_model_by_name, return_engine, get_all_ticker_strings, get_ticker_by_ticker, get_all_models, get_model_by_name, get_new_nasdaq_tickers
from data_api.db import get_model_by_id, model_name_exists, delete_all_models, load_formatted_train_data, create_ticker, delete_ticker_by_ticker
from data_api.init_db import initialize_db, update_price_data_sets, update_ticker_price_data
from models.model_func import save_model, load_model, make_predictions, convert_to_business_days
from helper_functions import empty_data_dirs, delete_model_files_not_in_db, delete_model_files
import models
import data_api

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
if os.path.exists(os.path.join(database_dir, db_filename)):
    # If the database already exits, get a connection to the database
    engine = return_engine(database_dir, db_filename=db_filename)
else:
    # Otherwise, create a new database and initialize it with data
    engine = initialize_db(database_dir,  db_filename=db_filename)

# Setup Flask
app = Flask(__name__)
app.config['TESTING'] = True
app.config['SECRET_KEY'] = '#$%^&*hf921th2023t348642tö02th23ß320'
socketio = SocketIO(app)


# API to load model parameters (required for select_model page)
@app.route("/get_model_params")
def get_model_params():
    try:
        model_name = request.args.get('model_name')
        model = get_model_by_name(engine, model_name)
        tickers = []
        for ticker_instance in model.tickers:
            tickers.append(ticker_instance.ticker)

        data = jsonify({"model_name":  model_name,
                        "start_date": model.start_date,
                        "end_date": model.end_date,
                        "tickers": tickers,
                        })
        return data
    except:
        data = jsonify({"model_name":  "",
                        "start_date": "",
                        "end_date": "",
                        "tickers": [""],
                        })
        return data


# Error messages in JSON
@app.errorhandler(werkzeug.exceptions.NotFound)
def notfound(e):
    return jsonify(error=str(e), mykey="myvalue"), e.code


# Grab all non-specified paths per default
@app.route("/", defaults={'path': ''})
@app.route("/<path:path>")
# Start Page
def index(path):
    app.logger.info("Opened start page")
    return render_template("index.html")


@app.route("/create_model", methods=['GET', 'POST'])
def create_model():
    tickers = get_all_ticker_strings(engine)
    # Open the page the first time
    if request.method == "GET":
        return render_template('create_model.html', tickers=tickers, created=False)

    # Submit the selected model data and create the model
    if request.method == "POST":
        # Specify the date format received by the POST request
        date_format = "%B %d, %Y"

        # Get the parameters from the POST request
        model_name = request.form.getlist('model_name')[0]

        if model_name_exists(engine, model_name):
            return render_template('create_model.html', tickers=tickers, notification_message="The model name \"{model_name}\" already exists, please select a different one", model_name=model_name)

        model_tickers = request.form.getlist('model_tickers')
        start_date = datetime.strptime(request.form['start_date'], date_format)
        end_date = datetime.strptime(request.form['end_date'], date_format)

        # Convert the tickers into ticker_ids
        model_tickers_ids = []
        for ticker in model_tickers:
            model_tickers_ids.append(
                get_ticker_by_ticker(engine, ticker).Ticker.id)

        # Load the training data
        try:
            train_data = load_formatted_train_data(
                engine, model_tickers_ids, start_date, end_date)
        except ValueError as value_error:
            print(value_error)
            print(f"The model could not be created: {value_error}")
            return render_template('create_model.html', tickers=tickers, notification_message=f"The model {model_name} could not be created: {value_error}", model_name=model_name)

        # If na values are used to train the model, all predictions will also be NA
        # If data is missing here, the reason is usually that the company was not public/the index did not exist before a certain date
        # Therefore we assume a hypothetical price of 0
        train_data.fillna(value=0, inplace=True)

        # Create a Keras ML model
        try:
            model, last_data, scaler, data_columns = models.model_func.create_model(
                train_data)
        except AssertionError as assertion_error:
            error_message = f"The model {model_name} could not be created: {assertion_error}"
            print(error_message)
            return render_template('create_model.html', tickers=tickers, notification_message=error_message, model_name=model_name)

        # Store the model to disk
        model_path = save_model(model, model_name)

        # Store the scaler to disk
        scaler_path = f"./data/scalers/{model_name}"
        joblib.dump(scaler, scaler_path)

        try:
            model_id = data_api.db.create_model(engine, model_name, start_date, end_date,
                                                model_tickers_ids, data_columns, last_data, scaler_path, model_path)
        except sqlalchemy.exc.InvalidRequestError as invalidRequestError:
            error_message = f"An invalid request error occured: {invalidRequestError}"
            print(error_message)
            return render_template('create_model.html', tickers=tickers, notification_message=error_message, model_name=model_name)

        # Store the model_id to the session
        session["model_id"] = model_id
        print(
            f"Model \"{model_name}\" with id {model_id} was successfully created!")

        return redirect(url_for('predict', model_tickers=tickers, start_date=start_date, end_date=end_date))
    else:
        return "Error, only GET and POST requests are supported"


@app.route("/select_model", methods=['GET', 'POST'])
def select_model():
    model_name_list = []

    models = get_all_models(engine)

    for model in models:
        model_name_list.append(model[0].model_name)

    if request.method == "GET":
        return render_template("select_model.html", model_name_list=model_name_list, selected=False, notification_message=None)

    if request.method == "POST":
        if request.form['submit_button'] == 'Delete':
            model_name = request.form.getlist('select_model')[0]
            model_name_list.remove(model_name)
            delete_model_by_name(engine, model_name)
            delete_model_files(model_name, model_dir, scaler_dir)
            return render_template("select_model.html", model_name_list=model_name_list, selected=False, notification_message=f"The model {model_name} has been deleted")

        elif request.form['submit_button'] == 'Select':
            model_name = request.form.getlist('select_model')[0]
            model = get_model_by_name(engine, model_name)
            tickers = []
            for ticker in model.tickers:
                tickers.append(ticker.ticker)
            session["model_id"] = model.id
            return redirect(url_for('predict', model_tickers=tickers, start_date=model.start_date, end_date=model.end_date))
        else:
            return "Error, there is no action specified for this submit button"
    else:
        return "Error, only GET and POST requests are supported"


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if session.get("model_id") != None:
        model = get_model_by_id(engine, session["model_id"])
        tickers = []
        for ticker in model.tickers:
            tickers.append(ticker.ticker)
        return render_template('predict.html', model_tickers=tickers, start_date=model.start_date, end_date=model.end_date)
    # No POST method (with parameters) or no parameters set
    else:
        return render_template('prediction_missing_values.html')


@app.route("/results", methods=['GET', 'POST'])
def results():
    if request.method == "POST":
        # Extract the tickers to predict from the form
        target_tickers = request.form.getlist('target_tickers')
        # Get the ticker ids
        ticker_ids = []
        # Create the column names to extract later
        result_col_names = []
        # Also create the column names to output for the user
        output_col_names = []
        for ticker in target_tickers:
            ticker_id = get_ticker_by_ticker(engine, ticker).Ticker.id
            ticker_ids.append(ticker_id)
            result_col_names.append(f"adj_close-{ticker_id}")
            output_col_names.append(f"Price for {ticker}")
        # Extract the dates to predict from the form
        dates = request.form.getlist('dates[]')
        # Convert the dates
        date_format = "%B %d, %Y"
        date_objs = []
        for date_str in dates:
            date_objs.append(datetime.strptime(date_str, date_format).date())
        date_objs = convert_to_business_days(date_objs)

        # Load the model data from the db
        model_db = get_model_by_id(engine, session["model_id"])
        scaler = joblib.load(model_db.scaler_path)
        # Load the model object from the filesystem
        model = load_model(model_db.model_name)

        # Calculate the date range which needs to be predicted
        date_range = pd.bdate_range(
            model_db.end_date.date(), max(date_objs) + pd.DateOffset(1))
        num_days = len(date_range)

        # Create a dataframe containing the predictions
        df_forecast = make_predictions(
            model, model_db.end_date, model_db.last_data, scaler, num_days, model_db.data_columns)

        # Only filter for the specified columns
        filtered_df = df_forecast[result_col_names]
        # Reformat the column names
        filtered_df.columns = output_col_names
        # Set the index to the correct dates
        filtered_df.index = date_range
        # Filter to only return the requested dates
        filtered_df = filtered_df.loc[date_objs]

        return render_template('results.html', predictions=filtered_df.to_html())
    # No POST method (with parameters) or no parameters set
    else:
        return render_template('results_missing_values.html')


@app.route("/edit_tickers", methods=['GET', 'POST'])
def edit_tickers():
    # Change to "asset" to view the stock names instead of the tickers in the GUI
    view_type = "ticker"

    # Download the available tickers
    all_tickers, asset_names = get_new_nasdaq_tickers(engine)
    # If the page is called using a GET method, show the page without any action except showing all available and existing tickers
    if request.method == "GET":
        existing_tickers = get_all_ticker_strings(engine)
        if view_type == "ticker":
            return render_template('edit_tickers.html', asset_names=all_tickers, existing_tickers=existing_tickers,  notification_message=None)
        elif view_type == "asset":
            return render_template('edit_tickers.html', asset_names=asset_names, existing_tickers=existing_tickers,  notification_message=None)
    elif request.method == "POST":
        if "add_tickers" in request.form:
            assets = request.form.getlist('asset_names')
            # Track which ones can be updated/downloaded and which ones not -> any error with the connection, the data source etc.
            successes = list()
            failures = list()
            for asset in assets:
                if view_type == "ticker":
                    ticker_index = all_tickers.index(asset)
                elif view_type == "asset":
                    ticker_index = asset_names.index(asset)
                ticker_id = create_ticker(
                    engine, ticker=all_tickers[ticker_index], name=asset_names[ticker_index])
                result = update_ticker_price_data(engine, ticker_id)
                if view_type == "ticker":
                    if result:
                        successes.append(all_tickers[ticker_index])
                    else:
                        failures.append(all_tickers[ticker_index])
                elif view_type == "asset":
                    if result:
                        successes.append(asset_names[ticker_index])
                    else:
                        failures.append(asset_names[ticker_index])
                # Update the ticker list so that the newly added tickers are included
                existing_tickers = get_all_ticker_strings(engine)

            notification_message = f"The following tickers have been added: {successes} \n \n The following tickers have not been added due to an error: {failures}"

            if view_type == "ticker":
                return render_template('edit_tickers.html', asset_names=all_tickers, existing_tickers=existing_tickers,  notification_message=notification_message)
            elif view_type == "asset":
                return render_template('edit_tickers.html', asset_names=asset_names, existing_tickers=existing_tickers,  notification_message=notification_message)

        elif "delete_tickers" in request.form:
            tickers = request.form.getlist('existing_tickers')
            # Delete the tickers and the linked models
            for ticker in tickers:
                delete_ticker_by_ticker(engine, ticker)
            # Delete the model files
            delete_model_files_not_in_db(engine, model_dir, scaler_dir)
            # Update the ticker list so that the newly deleted tickers are not included anymore
            existing_tickers = get_all_ticker_strings(engine)

            notification_message = f"The following tickers have been deleted: {tickers}"
            return render_template('edit_tickers.html', all_tickers=all_tickers, asset_names=asset_names, existing_tickers=existing_tickers,  notification_message=notification_message, added=False)
        else:
            return "Error"
    else:
        return "Error, only GET and POST requests are supported"


@app.route("/config", methods=['GET', 'POST'])
def config():
    try:
        if request.method == 'POST':
            if "reset_db" in request.form:
                empty_data_dirs(model_dir, scaler_dir)
                initialize_db(database_dir)
                return render_template('config.html', notification_message="Database reseted successfully")
            elif "update_db" in request.form:
                update_price_data_sets(engine)
                return render_template('config.html', notification_message="Database updated successfully")
            elif "delete_models" in request.form:
                empty_data_dirs(model_dir, scaler_dir)
                delete_all_models(engine)
                session["model_id"] = None
                return render_template('config.html', notification_message="All models have been deleted")
            else:
                return render_template('config.html', notification_message="Error")
        elif request.method == 'GET':
            return render_template('config.html',  notification_message=None)
    except sqlalchemy.exc.IntegrityError as err:
        print(err)
        return render_template('config.html',  notification_message=f"A database Integrity error occured: \n \n {err}")


@app.route("/about")
def about():
    return render_template('about.html')


if __name__ == '__main___':
    # threaded = True -> Automatically create a new thread for every session/user
    # app.run(port=5000, debug=False, threaded=True)
    socketio.run(port=5000, debug=False, threaded=True)
