import logging
from flask import Flask, session, render_template, redirect, sessions, url_for, request, make_response, escape, jsonify, Response, abort, Request
import requests
from tensorflow.python.keras.utils.generic_utils import default
from tensorflow.python.keras.utils.generic_utils import default
import numpy as np

from data_api.db import initialize_db, return_engine, get_ticker_strings
from datetime import datetime as dt

from models.model_func import create_model, save_model

engine = return_engine()


app = Flask(__name__)
app.config['TESTING'] = True
app.config['SECRET_KEY'] = '#$%^&*hf921th2023t348642tö02th23ß320'

# Logging
# Add "filename='record.log'" to log into a file
logging.basicConfig( level=logging.INFO, format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')
app.logger.info('Info level log')
app.logger.warning('Warning level log')


# Error messages in JSON
import werkzeug
@app.errorhandler(werkzeug.exceptions.NotFound)
def notfound(e):
    return jsonify(error=str(e), mykey = "myvalue"), e.code


# @app.route("/delete_input")
# def delete_input():
#     session.pop('name', None) # Cookies are a dictionary
#     return redirect(url_for('index'))


# Grab all non-specified paths per default
@app.route("/",defaults= {'path' : ''})
@app.route("/<path:path>")
def index(path):
    tickers = get_ticker_strings(engine)
    return render_template('index.html', tickers = tickers)
    

@app.route('/model', methods=['GET','POST'])
def model():
    if request.method == "POST":
        # Assign the data from the form to the session to make it accessible when needed
        session['model_name'] = request.form.getlist('model_name')
        session['model_tickers'] = request.form.getlist('model_tickers')
       # session['model_target'] = request.form['model_target']
        session['startdate'] = request.form['startdate']
        session['enddate'] = request.form['enddate']


        model, last_data, scaler = create_model(session['model_tickers'], session['startdate'], session['enddate'])

        # Create a model name
        # modelname = create_modelname(session['model_tickers'], session['startdate'], session['enddate'])
        modelname = session["modelname"]
        app.logger.info(f"Model {modelname} successfully created")
        # Store the model
        save_model(model, modelname)       

        # print("scaler.__dict__")
        # print(scaler.__dict__)
        # print("vars(scaler)")
        # print(vars(scaler))

        # for key, value in scaler.__dict__:
        #     print(f"Key: {key}; Value: {value}, Datatype of value: {type(value)} \n")
        # for key in scaler.__dict__:
        #     print(f"Key: {key} \n")
        for key in dir(scaler):
            print(f"Key: {key} \n")

        # session["scaler"] = vars(scaler)
        # session["scaler"] = scaler.__dict__

        # print("last_data")
        # print(last_data)
        # print("last_data.shape")
        # print(last_data.shape)

        # Encode last_data to json, so that we can store it in the Flask session
        # session["last_data"] = encode_array(last_data)
      
        return render_template('model.html', model_tickers= session['model_tickers'], model_target = session['model_target'], startdate = request.form['startdate'], enddate = request.form['enddate'])
    else:
        # Check if values are set
        return "only POST is supported"


@app.route("/test")
def test():
    abort(404)
    return "Hallo"


@app.route("/prediction", methods=['GET','POST'])
def prediction():
    if request.method == "POST":

        # print("request.form.getlist('num_days')")
        # print(request.form.getlist('num_days'))
        session['num_days'] = int(request.form.getlist('num_days')[0])
        model = load_model(session["modelname"])

        # last_data = decode_array(session["last_data"])
        # last_data = np.reshape(last_data, (1, last_data.shape[0], last_data.shape[1]))

        df_forecast = make_predictions(model, session["enddate"], last_data, scaler, session['num_days'], session['model_tickers'])
        # df_forecast_new = "Dummy"
        return render_template('prediction.html', df_forecast_new = df_forecast)
    else:
        # Check if values are set
        return "only POST is supported"


@app.route("/config")
def config():
    return "Config page"


@app.route("/about")
def about():
    return render_template('about.html')


if __name__ == '__main___':
    # threaded = True -> Automatically create a new thread for every session/user
    app.run(port=5000, debug=True, threaded = True)
    # TLS
    # app.run(port=5000, debug=False, ssl_context =context)

# conda activate uc-stock-price-pred
# export FLASK_ENV=development
# flask run