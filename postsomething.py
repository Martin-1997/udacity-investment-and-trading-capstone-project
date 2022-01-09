import requests
import json

url = "http://127.0.0.1:5000/postme/"
data = {'data' : 'lolBOBO'}

result = requests.post(url, json.dumps(data))

print("Data posted!")