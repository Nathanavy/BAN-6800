from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Create Flask app
app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Flask app is running!"

@app.route("/predict", methods=["GET", "POST"])
def predict():
    data = request.get_json()
    # Example dummy prediction
    return jsonify({"received_data": data, "prediction": 1})
    
# Load data
df = pd.read_csv('Walmart.csv')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
print('Shape:', df.shape)
df.head()


# Basic cleaning and features
df.sort_values(['Date'], inplace=True)

# Time-based features
df['Year'] = df['Date'].dt.year
df['DayOfWeek'] = df['Date'].dt.dayofweek

# If there's no "IsHoliday" column, create it as 0
if 'IsHoliday' in df.columns:
    df['IsHoliday'] = df['IsHoliday'].astype(int)
elif 'Holiday_Flag' in df.columns:   # your file has Holiday_Flag
    df['IsHoliday'] = df['Holiday_Flag'].astype(int)
else:
    df['IsHoliday'] = 0

# Lag and rolling mean based only on Date ordering
df['lag_1'] = df['Weekly_Sales'].shift(1)
df['rolling_mean_4'] = df['Weekly_Sales'].shift(1).rolling(window=4, min_periods=1).mean()

# Drop rows with missing lag/rolling values
df = df.dropna(subset=['lag_1','rolling_mean_4'])

print('After features shape:', df.shape)
df.head()


# Train/test split (time-based)
unique_dates = np.sort(df['Date'].unique())
split_date = unique_dates[int(len(unique_dates)*0.9)]

train_df = df[df['Date'] <= split_date].copy()
test_df = df[df['Date'] > split_date].copy()

# Features that actually exist in your dataset
features = ['IsHoliday', 'Year', 'DayOfWeek', 'lag_1', 'rolling_mean_4']
target = 'Weekly_Sales'

X_train = train_df[features]; y_train = train_df[target]
X_test = test_df[features]; y_test = test_df[target]

# Baseline linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

# Random forest
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# Evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

print('LinearRegression ->', evaluate(y_test, lr_preds))
print('RandomForest ->', evaluate(y_test, rf_preds))






