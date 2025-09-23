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

print("Initial Shape:", df.shape)
print(df.head())

# 2. Data Cleaning

# Convert numeric fields
for col in ['Weekly_Sales','Fuel_Price','Temperature','CPI','Unemployment']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Handle Holiday_Flag as binary
if 'Holiday_Flag' in df.columns:
    df['Holiday_Flag'] = pd.to_numeric(df['Holiday_Flag'], errors='coerce').fillna(0).astype(int)
    df['Holiday_Flag'] = df['Holiday_Flag'].clip(0,1)

# Handle missing values → median imputation (by Store)
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if col != 'Store':
        df[col] = df.groupby('Store')[col].transform(lambda x: x.fillna(x.median()))
        df[col] = df[col].fillna(df[col].median())

print("Shape after cleaning:", df.shape)


# 3. Feature Engineering

df['day_of_week'] = df['Date'].dt.day_name()

df['month'] = df['Date'].dt.month
df['is_weekend'] = (df['Date'].dt.weekday >= 5).astype(int)

def month_to_season(m):
    if m in [12,1,2]:
        return "Winter"
    elif m in [3,4,5]:
        return "Spring"
    elif m in [6,7,8]:
        return "Summer"
    else:
        return "Fall"

df['season'] = df['month'].apply(month_to_season)

# Save cleaned dataset
#df.to_csv("walmart_clean.csv", index=False)


# 4. Exploratory Data Analysis (EDA)


# --- A. Summary Stats ---
print("\nSummary Statistics:")
print(df.describe())

# --- B. Weekly Sales Trend ---
if 'Weekly_Sales' in df.columns:
    ts = df.dropna(subset=['Date','Weekly_Sales']).set_index('Date').sort_index()
    weekly = ts['Weekly_Sales'].resample('W').sum()
    plt.figure(figsize=(10,5))
    plt.plot(weekly.index, weekly.values, label="Weekly Sales")
    plt.title("Weekly Sales Trend")
    plt.xlabel("Week")
    plt.ylabel("Sales")
    plt.legend()
    plt.show()

# --- C. Sales by Store ---
if 'Store' in df.columns and 'Weekly_Sales' in df.columns:
    top_store = df.groupby('Store')['Weekly_Sales'].sum().sort_values(ascending=False).head(20)
    plt.figure(figsize=(12,6))
    sns.barplot(x=top_store.index, y=top_store.values, palette="Blues_d")
    plt.title("Top 20 Stores by Sales")
    plt.xlabel("Store")
    plt.ylabel("Total Sales")
    plt.show()

# --- D. Sales by Month ---
if 'month' in df.columns and 'Weekly_Sales' in df.columns:
    month_sales = df.groupby('month')['Weekly_Sales'].sum()
    plt.figure(figsize=(8,5))
    month_sales.plot(kind='bar', color='orange')
    plt.title("Sales by Month")
    plt.xlabel("Month")
    plt.ylabel("Total Sales")
    plt.show()

# --- E. Holiday vs Non-Holiday Sales ---
if 'Holiday_Flag' in df.columns and 'Weekly_Sales' in df.columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='Holiday_Flag', y='Weekly_Sales', data=df)
    plt.title("Holiday vs Non-Holiday Sales")
    plt.xlabel("Holiday Flag (0 = No, 1 = Yes)")
    plt.ylabel("Weekly Sales")
    plt.show()

# --- F. Correlation Heatmap ---
plt.figure(figsize=(8,6))
sns.heatmap(df[['Weekly_Sales','Temperature','Fuel_Price','CPI','Unemployment']].corr(),
            annot=True, cmap='coolwarm', center=0)
plt.title("Correlation Heatmap")
plt.show()
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






