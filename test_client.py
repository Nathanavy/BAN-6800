import requests

url = "http://54.167.76.137:5000/predict"
data = {"feature1": 10, "feature2": 5}

resp = requests.post(url, json=data)
print(resp.json())
