import requests

url = "http://127.0.0.1:5000/predict"
data = {
    "DayOfWeek": 3,
    "Holiday_Flag": 0,
    "Year": 2012,
    "lag_1": 20000,
    "rolling_mean_4": 21000
}

response = requests.post(url, json=data)
print(response.json())
