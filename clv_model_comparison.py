import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# Step 1: Simulate CLV as the target variable
def simulate_clv(rfm_df):
    """
    Simulates CLV (Customer Lifetime Value) based on the RFM metrics.

    Parameters:
        rfm_df (DataFrame): RFM metrics dataset.

    Returns:
        rfm_df (DataFrame): Updated dataset with a simulated 'CLV' column.
    """
    # Simulate CLV based on Frequency and Monetary with some noise
    rfm_df["CLV"] = rfm_df["Frequency"] * rfm_df["Monetary"] * np.random.uniform(0.8, 1.2, len(rfm_df))
    return rfm_df

# Step 2: Train-test split
def prepare_data_for_modeling(rfm_df):
    """
    Prepares data for training and testing the CLV prediction model.

    Parameters:
        rfm_df (DataFrame): Dataset with RFM metrics and simulated CLV.

    Returns:
        X_train, X_test, y_train, y_test: Split data for model training and testing.
    """
    X = rfm_df[["Recency", "Frequency", "Monetary"]]
    y = rfm_df["CLV"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train and evaluate models
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Trains and evaluates multiple ML models for CLV prediction.

    Parameters:
        X_train (DataFrame): Training features.
        X_test (DataFrame): Test features.
        y_train (Series): Training target.
        y_test (Series): Test target.

    Returns:
        results (DataFrame): Performance metrics of all models.
    """
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "XGBoost": XGBRegressor(objective="reg:squarederror", random_state=42)
    }

    metrics_list = []

    for model_name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Calculate evaluation metrics
        metrics = {
            "Model": model_name,
            "R2 Score": r2_score(y_test, predictions),
            "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
            "MAE": mean_absolute_error(y_test, predictions)
        }
        metrics_list.append(metrics)

    # Convert metrics to a DataFrame
    results = pd.DataFrame(metrics_list)
    return results

# Step 4: Execute the steps
if __name__ == "__main__":
    # Load RFM data
    rfm_df = pd.read_csv("rfm_metrics.csv")

    # Simulate CLV
    rfm_df = simulate_clv(rfm_df)

    # Split data for modeling
    X_train, X_test, y_train, y_test = prepare_data_for_modeling(rfm_df)

    # Train and evaluate models
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # Save results to a CSV file
    results.to_csv("model_comparison_metrics.csv", index=False)

    # Print results
    print("Model Comparison Metrics:")
    print(results)