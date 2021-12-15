from flask import Flask, session, render_template, redirect, sessions
from flask import render_template
from tensorflow.python.keras.utils.generic_utils import default
from backend.access_api import get_tickers
from backend.model import get_model, make_predictions
from datetime import date, timedelta
from flask_wtf import FlaskForm
from tensorflow.python.keras.utils.generic_utils import default
from wtforms.fields import DateField
from wtforms.fields.numeric import IntegerField
from wtforms.validators import DataRequired
from wtforms import validators, SubmitField
from wtforms.widgets.core import NumberInput
from wtforms.fields.choices import SelectField, SelectMultipleField


app = Flask(__name__)
app.config['SECRET_KEY'] = '#$%^&*'

tickers, names = get_tickers()
end_date = date.today()
start_date = end_date - timedelta(days = 100)

class CreateModelForm(FlaskForm):
    startdate = DateField('Start Date',format='%Y-%m-%d', default = date.today() - timedelta(days = 100), validators=(validators.DataRequired(),)) # 
    enddate = DateField('End Date',format='%Y-%m-%d', default = date.today(), validators=(validators.DataRequired(),))
    model_tickers = SelectMultipleField('Model tickers',  validators=(validators.DataRequired(),)) 
    submit = SubmitField('Create Model')

class PredictionForm(FlaskForm):
    n_days = IntegerField("Days to predict", default = 1, widget = NumberInput(min = 1, max = 100))
    ticker_predict = SelectField("Ticker to predict", choices = tickers, default = tickers[0]) # [tickers[0], tickers[3]]) #, choices = tickers_predict, default = tickers_predict_default)
    submit = SubmitField('Query Model')

@app.route("/", methods=['GET','POST'])
def index():
    form = CreateModelForm()
   # form.model_tickers.default = [tickers[0], tickers[3]]
   # form.process()

    # ui fluid search selection 

    if form.validate_on_submit():
        session['startdate'] = form.startdate.data
        session['enddate'] = form.enddate.data
        session['model_tickers'] = form.model_tickers.data
        return redirect('model')
    return render_template('index.html', tickers = tickers, form=form)


@app.route('/model', methods=['GET','POST'])
def model():
    start_date = session['startdate']
    end_date = session['enddate']
    model_tickers = session['model_tickers']

    model = get_model(model_tickers, start_date, end_date)

    prediction_form = PredictionForm(1, model_tickers, model_tickers[0])
    if prediction_form.validate_on_submit():
        session['model'] = model
        session['n_days'] = prediction_form.n_days.data
        session['ticker_predict'] = prediction_form.ticker_predict.data
        return redirect('prediction')
    return render_template('model.html', start_date = start_date, end_date = end_date, model_tickers = model_tickers, form = prediction_form)



@app.route("/prediction", methods=['GET','POST'])
def prediction():
    model = session['model']
    n_days = session['n_days']
    ticker_predict = session['ticker_predict']

    last_date_from_data, combined_dfs, original, df_forecast_new = make_predictions(model, n_days)
    return render_template('prediction.html')



@app.route("/about")
def about():
    return render_template('about.html')
