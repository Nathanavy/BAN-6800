import requests

url = "http://127.0.0.1:5000/predict"
data = {"feature1": 10, "feature2": 5}

resp = requests.post(url, json=data)
print(resp.json())
