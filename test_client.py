import json
from app import app

def test_predict():
    client = app.test_client()
    response = client.post(
        "/predict",
        json={
            "DayOfWeek": 3,
            "Holiday_Flag": 0,
            "Year": 2012,
            "lag_1": 20000,
            "rolling_mean_4": 21000
        }
    )
    data = response.get_json()
    assert "prediction" in data
    assert isinstance(data["prediction"], float)
