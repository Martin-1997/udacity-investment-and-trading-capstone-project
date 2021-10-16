try:
    from flask import Flask, render_template
    from markupsafe import escape
    import json
    import requests
except Exception as e:
    print("Error: {}".format(e))


app = Flask(__name__)


@app.route("/")
def home():
    return render_template('index.html')

