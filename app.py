# Import Python libraries
import werkzeug
import logging
from flask import Flask, session, render_template, sessions, request, jsonify, Request
from tensorflow.python.keras.utils.generic_utils import default
import numpy as np
from datetime import datetime as dt
import joblib
import os


# Import own libraries
from data_api.db import initialize_db, return_engine, get_ticker_strings, load_ticker_by_ticker, update_price_data_sets
from models.model_func import create_model, save_model, load_train_data, load_model, make_predictions


# Get a connection to the database
engine = return_engine()

# Define the data paths
current_dir = os.path.dirname(__file__)
model_dir = os.path.join(current_dir, "data/models")
scaler_dir = os.path.join(current_dir, "data/scalers")
last_data_dir = os.path.join(current_dir, "data/last_data")
data_columns_dir = os.path.join(current_dir, "data/data_columns")


# Setup Flask
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
    app.logger.info("Opened start page")
    tickers = get_ticker_strings(engine)
    return render_template('index.html', tickers=tickers)


@app.route("/select_model")
def select_model():
    app.logger.info("Opened select model page")
    print(model_dir)
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            print(file)

    scaler_dir = os.path.join(current_dir, "data/scalers")
    for root, dirs, files in os.walk(scaler_dir):
        for file in files:
            print(file)

    last_data_dir = os.path.join(current_dir, "data/last_data")
    for root, dirs, files in os.walk(last_data_dir):
        for file in files:
            print(file)
            
    return render_template("select_model.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Check if the page is opened with a POST request (new data is entered) and if the required values are already set
    parameters_set = (
        ('modelname' in session) 
        and ('model_tickers' in session)
        and ('startdate' in session) 
        and ('enddate' in session) 
        )
    POST_method = (request.method == "POST")
    if POST_method or parameters_set:
        if POST_method:
            # Assign the data from the form to the session to make it accessible when needed
            session['modelname'] = request.form.getlist('model_name')[0]
            session['model_tickers'] = request.form.getlist('model_tickers')
            session['startdate'] = request.form['startdate']
            session['enddate'] = request.form['enddate']

        # Load model, if model exists
        if model_exists(session['modelname']):
            session['model_path'] = os.path.join(model_dir, f"{session['modelname']}.5")
            session['scaler_path'] = os.path.join(scaler_dir, f"{session['modelname']}")
            session["last_data_path"] = os.path.join(last_data_dir, f"{session['modelname']}.npy")
            session["data_columns"] = []
            with open(os.path.join(data_columns_dir, session['modelname']), "r") as f:
                for line in f:
                    session["data_columns"].append(str(line.strip()))
            loaded = True
        else:
            train_data = load_train_data(
                engine, session['model_tickers'], session['startdate'], session['enddate'])
            # If na values are used to train the model, all predictions will also be NA
            train_data.fillna(value=0, inplace=True)
            model, last_data, scaler, data_columns = create_model(train_data)

            session["data_columns"] = data_columns.tolist()
            # Store the data columns to a file
            with open(os.path.join(data_columns_dir, session['modelname']), "w") as f:
                for column in session["data_columns"]:
                    f.write(str(column) +"\n")

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
            loaded = False
        return render_template('predict.html', model_tickers=session['model_tickers'], model_path=session['model_path'], startdate = session['startdate'], enddate = session['enddate'],loaded = loaded)
    # No POST method (with parameters) or no parameters set   
    else:
        return render_template('prediction_missing_values.html')
        

@app.route("/results", methods=['GET', 'POST'])
def results():
    # Check if the page was called by a POST method and if the parameters have been already set
    parameter_set = (
            ('modelname' in session) 
            and ('scaler_path' in session) 
            and ('last_data_path' in session)  
            and ('data_columns' in session) 
            and ('modelname' in session) 
            and ('model_tickers' in session)
            and ('startdate' in session)  
            and ('enddate' in session)
            and ('num_days' in session) #(len(request.form.getlist('num_days')) > 0) 
            and ('target_ticker' in session)    # (len(request.form.getlist('target_ticker')) > 0)
            )
    POST_method = (request.method == "POST")

    if POST_method or parameter_set:
        # If POST method: Set the parameters
        if POST_method:
            session['num_days'] = int(request.form.getlist('num_days')[0])
            session['target_ticker'] = request.form.getlist('target_ticker')[0]

        ticker_id = load_ticker_by_ticker(
            engine, session['target_ticker']).Ticker.id
        model = load_model(session["modelname"])
        scaler = joblib.load(session['scaler_path'])
        last_data = np.load(session["last_data_path"])

        df_forecast = make_predictions(
            model, session["enddate"], last_data, scaler, session['num_days'], session["data_columns"])
        # only forward the forcast data for the correct ticker
        result_col_name = "adj_close-" + str(ticker_id)
        return render_template('results.html', predictions=df_forecast[result_col_name])
    # No POST method (with parameters) or no parameters set
    else:
        return render_template('results_missing_values.html')


def model_exists(modelname):
    """
    Checks if a model (and the additional required data) with a given name already exits
    """
    model_flag = os.path.isfile(os.path.join(model_dir, f"{modelname}.h5"))
    scaler = os.path.isfile(os.path.join(scaler_dir, f"{modelname}"))
    last_data = os.path.isfile(os.path.join(last_data_dir, f"{modelname}.npy"))
    data_columns = os.path.isfile(os.path.join(data_columns_dir, f"{modelname}"))

    return (model_flag and scaler and last_data and data_columns)


def empty_data_dirs():
    """
    This method delets all files in the last_data, models and scalers-folders
    """
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
        return render_template('config.html', information = "")
        

@app.route("/about")
def about():
    return render_template('about.html')


if __name__ == '__main___':
    # threaded = True -> Automatically create a new thread for every session/user
    app.run(port=5000, debug=False, threaded=True)

# conda activate uc-stock-price-pred
# export FLASK_ENV=development
# flask run
