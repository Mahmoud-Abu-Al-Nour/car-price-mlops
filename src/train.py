#Import the libraries
import pandas as pd
import numpy as np
import pickle
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
X_train = pd.read_csv('data/X_train.csv')
X_test = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').squeeze()
y_test = pd.read_csv('data/y_test.csv').squeeze()

# Set MLflow experiment
mlflow.set_experiment("car-price-prediction")

def evaluate(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return mae, rmse, r2

# Experiment 1 — Random Forest
with mlflow.start_run(run_name="Random Forest"):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae, rmse, r2 = evaluate(y_test, y_pred)
    
    # Log parameters
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    
    # Log metrics
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    print(f"Random Forest → MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.2f}")

# Experiment 2 — Linear Regression
with mlflow.start_run(run_name="Linear Regression"):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae, rmse, r2 = evaluate(y_test, y_pred)
    
    # Log parameters
    mlflow.log_param("model_type", "LinearRegression")
    
    # Log metrics
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    print(f"Linear Regression → MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.2f}")

print("\nAll experiments logged successfully!")