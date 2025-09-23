from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import traceback

# ==========================
# Flask setup
# ==========================
app = Flask(__name__)
CORS(app)

# ==========================
# Data Loading & Preprocessing
# ==========================
df = pd.read_csv("Walmart.csv")
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

print("Initial Shape:", df.shape)

# Convert numeric fields safely
for col in ["Weekly_Sales", "Fuel_Price", "Temperature", "CPI", "Unemployment"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Handle Holiday_Flag as binary
if "Holiday_Flag" in df.columns:
    df["Holiday_Flag"] = pd.to_numeric(df["Holiday_Flag"], errors="coerce").fillna(0).astype(int)
    df["Holiday_Flag"] = df["Holiday_Flag"].clip(0, 1)
else:
    df["Holiday_Flag"] = 0

# Fill missing numeric values → median per Store
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if col != "Store":
        df[col] = df.groupby("Store")[col].transform(lambda x: x.fillna(x.median()))
        df[col] = df[col].fillna(df[col].median())

print("Shape after cleaning:", df.shape)

# ==========================
# Feature Engineering
# ==========================
df = df.sort_values("Date").copy()
df["Year"] = df["Date"].dt.year
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["lag_1"] = df["Weekly_Sales"].shift(1)
df["rolling_mean_4"] = df["Weekly_Sales"].shift(1).rolling(window=4, min_periods=1).mean()
df = df.dropna(subset=["lag_1", "rolling_mean_4"])

# ==========================
# Train/Test Split
# ==========================
unique_dates = np.sort(df["Date"].unique())
split_date = unique_dates[int(len(unique_dates) * 0.9)]
train_df = df[df["Date"] <= split_date].copy()
test_df = df[df["Date"] > split_date].copy()

features = ["Holiday_Flag", "Year", "DayOfWeek", "lag_1", "rolling_mean_4"]
target = "Weekly_Sales"

X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]

# ==========================
# Model Training
# ==========================
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# Evaluation function
def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

print("LinearRegression ->", evaluate(y_test, lr_preds))
print("RandomForest ->", evaluate(y_test, rf_preds))

# ==========================
# Flask Routes
# ==========================
@app.route("/")
def home():
    return "Flask app is running!."

@app.route("/predict", methods=["GET", "POST"])
def predict():
    try:
        # Default (GET) input
        if request.method == "GET":
            data = {
                "Holiday_Flag": 0,
                "Year": 2012,
                "DayOfWeek": 3,
                "lag_1": 20000,
                "rolling_mean_4": 21000
            }
        else:
            # POST JSON input
            if not request.is_json:
                return jsonify({"error": "Request must be JSON"}), 400
            data = request.get_json()

        print("DEBUG: Received data ->", data)

        # Ensure all features exist
        defaults = {"Holiday_Flag": 0, "Year": 2012, "DayOfWeek": 0, "lag_1": 0, "rolling_mean_4": 0}
        input_data = np.array([[data.get(f, defaults[f]) for f in features]])

        # Predict with trained RF model
        prediction = rf.predict(input_data)[0]

        return jsonify({
            "received_data": data,
            "prediction": float(prediction)
        })

    except Exception as e:
        print("ERROR in /predict:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# ==========================
# Run Flask App
# ==========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
