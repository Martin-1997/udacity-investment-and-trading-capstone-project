import werkzeug
import logging
from flask import Flask, session, render_template, sessions, request, jsonify, Request
from tensorflow.python.keras.utils.generic_utils import default
import numpy as np
from datetime import datetime as dt
import joblib
import os

from data_api.db import initialize_db, return_engine, get_ticker_strings, load_ticker_by_ticker, update_price_data_sets
from models.model_func import create_model, save_model, load_train_data, load_model, make_predictions

engine = return_engine()

app = Flask(__name__)
app.config['TESTING'] = True
app.config['SECRET_KEY'] = '#$%^&*hf921th2023t348642tö02th23ß320'

# Logging
# Add "filename='record.log'" to log into a file
logging.basicConfig(level=logging.INFO,
                    format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')
app.logger.info('Info level log')
app.logger.warning('Warning level log')


# Error messages in JSON
@app.errorhandler(werkzeug.exceptions.NotFound)
def notfound(e):
    return jsonify(error=str(e), mykey="myvalue"), e.code


# Grab all non-specified paths per default
@app.route("/", defaults={'path': ''})
@app.route("/<path:path>")
def index(path):
    tickers = get_ticker_strings(engine)
    return render_template('index.html', tickers=tickers)


@app.route('/model', methods=['GET', 'POST'])
def model():
    if request.method == "POST":
        # Assign the data from the form to the session to make it accessible when needed
        session['modelname'] = request.form.getlist('model_name')[0]
        session['model_tickers'] = request.form.getlist('model_tickers')
        session['startdate'] = request.form['startdate']
        session['enddate'] = request.form['enddate']

        train_data = load_train_data(
            engine, session['model_tickers'], session['startdate'], session['enddate'])
        # If na values are used to train the model, all predictions will also be NA
        train_data.fillna(value=0, inplace=True)
        model, last_data, scaler, data_columns = create_model(train_data)
        session["data_columns"] = data_columns.tolist()

        # Create a model name
        app.logger.info(f"Model {session['modelname']} successfully created")
        # Store the model
        session['model_path'] = save_model(model, session["modelname"])

        #dirname = os.path.dirname(__file__)
        #print(f"dirname: {dirname}")
        # filename = dirname + f"/data/scaler/test"  # {session['modelname']}.joblib"
        #dump(scaler, f"./data/scalers/{'scaler_s'}")
        session['scaler_path'] = f"./data/scalers/{session['modelname']}"
        joblib.dump(scaler, session['scaler_path'])

        session["last_data_path"] = f"./data/last_data/{session['modelname']}"
        np.save(session["last_data_path"], last_data)

        return render_template('model.html', model_tickers=session['model_tickers'], model_path=session['model_path'], startdate=request.form['startdate'], enddate=request.form['enddate'])
    else:
        # Check if values are set
        return "only POST is supported"


@app.route("/prediction", methods=['GET', 'POST'])
def prediction():
    if request.method == "POST":
        session['num_days'] = int(request.form.getlist('num_days')[0])
        session['target_ticker'] = request.form.getlist('target_ticker')[0]
        ticker_id = load_ticker_by_ticker(
            engine, session['target_ticker']).Ticker.id
        model = load_model(session["modelname"])
        scaler = joblib.load(session['scaler_path'])
        last_data = np.load(session["last_data_path"] + ".npy")

        df_forecast = make_predictions(
            model, session["enddate"], last_data, scaler, session['num_days'], session["data_columns"])
        # only forward the forcast data for the correct ticker
        result_col_name = "adj_close-" + str(ticker_id)
        return render_template('prediction.html', predictions=df_forecast[result_col_name])
    else:
        # Check if values are set
        return "only POST is supported"

def empty_data_dirs():
    """
    This method delets all files in the last_data, models and scalers-folders
    """
    current_dir = os.path.dirname(__file__)
    model_dir = os.path.join(current_dir, "data/models")
    scaler_dir = os.path.join(current_dir, "data/scalers")
    last_data_dir = os.path.join(current_dir, "data/last_data")
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            os.remove(os.path.join(root, file))
    for root, dirs, files in os.walk(scaler_dir):
        for file in files:
            os.remove(os.path.join(root, file))
    for root, dirs, files in os.walk(last_data_dir):
        for file in files:
            os.remove(os.path.join(root, file))
    return True
        

@app.route("/config", methods=['GET', 'POST'])
def config():
    if request.method == 'POST':
        if "reset_db" in request.form:
            initialize_db()
            return render_template('config.html', information = "Database successfully reset")
        elif "update_db" in request.form:
            update_price_data_sets(engine)
            return render_template('config.html', information = "Database successfully updated")
        elif "delete_models" in request.form:
            empty_data_dirs()
            return render_template('config.html', information = f"All models have been deleted")
        else:
            return render_template('config.html', information = "error")
    elif request.method == 'GET':
        return render_template('config.html', information = "Test")
        


@app.route("/about")
def about():
    return render_template('about.html')


if __name__ == '__main___':
    # threaded = True -> Automatically create a new thread for every session/user
    app.run(port=5000, debug=False, threaded=True)

# conda activate uc-stock-price-pred
# export FLASK_ENV=development
# flask run
