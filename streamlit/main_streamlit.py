# Imports
import streamlit as st
from datetime import date, datetime
from plotly import graph_objects as go
import keras
from access_api import get_stock_data
from model import get_prediction

def main():
    
    st.title("Udacity Stock Price Predictor")
    try:
        start_date, end_date = collect_daterange()
        daterange = True
        print(f"Start date: {start_date}; End date: {end_date}")
    except:
        pass
    if start_date is not None and end_date is not None:
        try:
            model_tickers = collect_base_tickers()
            print(f"Model tickers: {model_tickers}")
            model_selection = True
        except:
            pass
    if model_tickers is not None:
        print("Ready to ask for prediction")
   
    #selected_stock, n_days = collect_prediction_data(model_tickers)
    #print(f"Selected stock: {selected_stock}; n days for prediction: {n_days}")


# Stocks available to our model
stocks = ["BTC-USD", "AAPL", "GOOG", 
        'GC=F', # Gold
      #  'GSPC', # S&P500
      #  'CL=F', # Crude Oil
      #  '^TNX', # 10 years US treasuries
        'SI=F', # Silver
        'EURUSD=X', # EUR-USD
        'MSFT',
        ]

# Methods
# We use an additional function here to make use of streamlits ability to cache data
@st.cache
def st_load_data(selected_stock, START, TODAY):
    data = get_stock_data(selected_stock, START, TODAY, date_index = False)
    return data

@st.cache
def st_get_prediction(tickers, target_var, start_date, end_date, n_days_for_prediction):
    last_date_from_data, combined_dfs, original, df_forecast = get_prediction(tickers, target_var, start_date, end_date, n_days_for_prediction)
    return last_date_from_data, combined_dfs, original, df_forecast

def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y = data["Adj Close"], name='adj_close'))
    fig.add_trace(go.Scatter(x=data["Date"], y = data["Open"], name='open'))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)



def collect_daterange():
    with st.form(key = "select_date_basis"):
        st.subheader("Select timerange for base data")
        TODAY = date.today()
        DEFAULT_START = date(year=2021, month = 1, day = 1)
        START = st.date_input(label = "Start date", value=DEFAULT_START, min_value=None, max_value=TODAY, key=None, help=None, on_change=None, args=None, kwargs=None)
        END = st.date_input(label = "End date", value=TODAY, min_value=START, max_value=TODAY, key=None, help=None, on_change=None, args=None, kwargs=None)
        # .strftime("%Y-%m-%d")
        submit_button = st.form_submit_button(label="Submit")
        if submit_button:
            return START, END

def collect_base_tickers():
    with st.form(key = "select_ticker_basis"):
        st.subheader("Select data to create model")
        model_tickers = st.multiselect('Select data to create model:', stocks) # .remove([selected_stock])
        model_basis_submit_button = st.form_submit_button(label="Submit")
        if model_basis_submit_button:
            return model_tickers

def collect_prediction_data(model_tickers):
    with st.form(key = "select_prediction_basis"):
        st.subheader("Select ticker to predict...")
        selected_stock = st.selectbox("Select stock to predict", model_tickers, index = 0)
        n_days = st.slider("Days to predict: ", 1, 100)
        pred_submit_button = st.form_submit_button(label="Predict")
        if pred_submit_button:
            return selected_stock, n_days

start_date, end_date = None, None
model_tickers = None  




main()