# Udacity Investment and Trading Capstone Project

# Table of Contents
1. [Overview](#Overview)
2. [Installation](#Installation)
3. [Usage](#Usage)
4. [Acknowledgements](Acknowledgements)
5. [Licence](Licence)

# Overview
Link to Udacity project proposal: https://docs.google.com/document/d/1ycGeb1QYKATG6jvz74SAMqxrlek9Ed4RYrzWNhWS-0Q/pub

This project contains a Flask Web-App which allows the user to predict future asset prices. The user can create a model by choosing the tickers and the time range which should be used to train a machine learning model in the background. Afterwards, specific (future) dates can be selected and the price for one or multiple of the assets from the training set can be predicted. Models are automatically stored and can be deleted, if they are no longer required. Additionally, the user can add new tickers from NASDAQ, if the price data is available at Yahoo Finance. Tickers and their linked price data can also be deleted. The price and trade volume data for the tickers is stored in a background database, so that training can be done offline and without a requirement to download the data every time. The price data can also be updated to include the latest available prices.

The following Python libraries need to be installed:
- flask
- Flask-SocketIO 
- jinja2
- flask_wtf
- wtforms
- flask-datepicker
- tensorflow
- keras
- joblib
- pandas_datareader
- sqlalchemy
- matplotlib
- sklearn

# Usage

The application is written using the Python web framework Flask. Additionally, Tensorflow and Keras are used for training the machine learning models and Sqlalchemy is used to manage the database for storing the required data. The file `app.py` contains the main code which controls the program flow and the different flask web pages. The following steps need to be performed to run the application in Flask development mode:

Activate flask development environment:
`export FLASK_ENV=development`

Run the flask application
`flask run`

Additionally, here you can find an overview of the remaining files and folders in the repository:

***./data_api:*** Classes and functions to access the Yahoo Finance API, the SQLite database and the SQLAlchemy database access framework. Here you can find a definition of the database scheme used in the background as well as different methods to send high level commands to the database (like initializing or updating the whole database) or low level commands to add, delete or edit entries.

***./models:*** Classes and functions which are required to use machine learning models for price data prediction as well as functions to preprocess the data.

***./templates:*** Templates for the web pages. These templates get loaded from the flask application and content gets filled in dynamically with jinja.

***./data:*** This directory gets automatically created when you start using the application to store the database, as well as models and required scaler objects to the disk.

***helper_functions.py:*** This file contains some helper functions to read and write data to the disk.

# Acknowledgements

I want to express gratitudes for the team at Udacity which provides a great course on the topic of Data Science.

# Licence

The code is available to anybody for personal and commercial use with the GNU General Public License v3.0. The used libraries may have there own different licences.
I do not guarantee any correct functionality of the code. Use it at your own risk, no warranty is provided.