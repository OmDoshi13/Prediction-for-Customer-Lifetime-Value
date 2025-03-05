import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_squared_error




def simulate_clv(rfm_df):
    """
    Simulates CLV (Customer Lifetime Value) based on the RFM metrics.

    Parameters:
        rfm_df (DataFrame): RFM metrics dataset.

    Returns:
        rfm_df (DataFrame): Updated dataset with a simulated 'CLV' column.
    """
    if "CLV" not in rfm_df.columns:
        # Simulate CLV based on Frequency and Monetary with some noise
        rfm_df["CLV"] = rfm_df["Frequency"] * rfm_df["Monetary"] * np.random.uniform(0.8, 1.2, len(rfm_df))
        print("CLV column has been simulated.")
    else:
        print("CLV column already exists.")
    return rfm_df

def train_best_model(rfm_df):
    """
    Trains the best-performing model on the full dataset.

    Parameters:
        rfm_df (DataFrame): RFM metrics dataset with target (CLV).

    Returns:
        model: Trained model.
    """
    # Prepare features and target
    X = rfm_df[["Recency", "Frequency", "Monetary"]]
    y = rfm_df["CLV"]

    # Split data for training and testing (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model (Random Forest Regressor as the best-performing model example)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    metrics = {
        "R2 Score": r2_score(y_test, predictions),
        "RMSE": root_mean_squared_error(y_test, predictions, squared=False),
        "MAE": mean_absolute_error(y_test, predictions)
    }
    print("Model Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    return model


def predict_clv(model, rfm_df):
    """
    Predicts CLV for all customers using the trained model.

    Parameters:
        model: Trained model.
        rfm_df (DataFrame): RFM metrics dataset.

    Returns:
        rfm_with_predictions (DataFrame): Dataset with predicted CLV values.
    """
    # Prepare features for prediction
    X = rfm_df[["Recency", "Frequency", "Monetary"]]

    # Predict CLV
    rfm_df["Predicted_CLV"] = model.predict(X)
    return rfm_df


if __name__ == "__main__":
    # Load the RFM dataset
    rfm_df = pd.read_csv("rfm_metrics.csv")

    # Simulate CLV if it's missing
    rfm_df = simulate_clv(rfm_df)

    # Train the best-performing model
    print("Training the best-performing model...")
    model = train_best_model(rfm_df)

    # Predict CLV for all customers
    print("Predicting CLV for all customers...")
    rfm_with_predictions = predict_clv(model, rfm_df)

    # Save the dataset with predictions to a CSV file
    rfm_with_predictions.to_csv("rfm_with_predictions.csv", index=False)
    print("Predictions saved to 'rfm_with_predictions.csv'.")