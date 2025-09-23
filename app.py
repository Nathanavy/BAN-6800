from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------------
# Flask setup
# -----------------------------
app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "✅ Flask app is running! Use POST /predict with JSON."

@app.route("/predict", methods=["GET", "POST"])
def predict():
    try:
        # If GET (browser visit), use some default input
        if request.method == "GET":
            data = {
                "IsHoliday": 0,
                "Year": 2012,
                "DayOfWeek": 3,
                "lag_1": 20000,
                "rolling_mean_4": 21000
            }
        else:
            # POST case: require JSON
            if not request.is_json:
                return jsonify({"error": "Request must be JSON"}), 400
            data = request.get_json()

        print("DEBUG: Received data ->", data)

        # Features
        features = ["IsHoliday", "Year", "DayOfWeek", "lag_1", "rolling_mean_4"]
        defaults = {"IsHoliday": 0, "Year": 2012, "DayOfWeek": 0, "lag_1": 0, "rolling_mean_4": 0}
        input_data = np.array([[data.get(f, defaults[f]) for f in features]])

        # Use your trained model (rf for example)
        prediction = rf.predict(input_data)[0]

        return jsonify({
            "received_data": data,
            "prediction": float(prediction)
        })

    except Exception as e:
        import traceback
        print("ERROR in /predict:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


        # Make prediction
        pred = rf.predict(input_data)[0]

        return jsonify({
            "received_data": data,
            "prediction": float(pred)
        })

    except Exception as e:
        import traceback
        print("ERROR in /predict:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# -----------------------------
# Model training (runs once at startup)
# -----------------------------
df = pd.read_csv("Walmart.csv")
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Sort and create features
df.sort_values(["Date"], inplace=True)
df["Year"] = df["Date"].dt.year
df["DayOfWeek"] = df["Date"].dt.dayofweek

if "IsHoliday" in df.columns:
    df["IsHoliday"] = df["IsHoliday"].astype(int)
elif "Holiday_Flag" in df.columns:
    df["IsHoliday"] = df["Holiday_Flag"].astype(int)
else:
    df["IsHoliday"] = 0

df["lag_1"] = df["Weekly_Sales"].shift(1)
df["rolling_mean_4"] = df["Weekly_Sales"].shift(1).rolling(window=4, min_periods=1).mean()
df = df.dropna(subset=["lag_1", "rolling_mean_4"])

# Train/test split
unique_dates = np.sort(df["Date"].unique())
split_date = unique_dates[int(len(unique_dates) * 0.9)]
train_df = df[df["Date"] <= split_date].copy()
test_df = df[df["Date"] > split_date].copy()

features = ["IsHoliday", "Year", "DayOfWeek", "lag_1", "rolling_mean_4"]
target = "Weekly_Sales"

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

# Train models
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# Print evaluation (for server logs)
def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

print("LinearRegression ->", evaluate(y_test, lr_preds))
print("RandomForest ->", evaluate(y_test, rf_preds))

# -----------------------------
# Run Flask (for local testing)
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


