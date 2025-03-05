import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_metrics(results_file):
    """
    Visualizes model comparison metrics using bar charts.

    Parameters:
        results_file (str): Path to the CSV file containing model evaluation metrics.
    """
    # Load results
    results = pd.read_csv(results_file)

    # Plot R2 Scores
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Model", y="R2 Score", data=results, hue="Model", dodge=False, palette="Blues_d", legend=False)
    plt.title("Model Comparison: R2 Score", fontsize=16)
    plt.ylabel("R2 Score")
    plt.xlabel("Model")
    plt.xticks(rotation=45)
    plt.grid(axis="y", alpha=0.75)
    plt.show()

    # Plot RMSE
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Model", y="RMSE", data=results, hue="Model", dodge=False, palette="Greens_d", legend=False)
    plt.title("Model Comparison: RMSE", fontsize=16)
    plt.ylabel("RMSE")
    plt.xlabel("Model")
    plt.xticks(rotation=45)
    plt.grid(axis="y", alpha=0.75)
    plt.show()

    # Plot MAE
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Model", y="MAE", data=results, hue="Model", dodge=False, palette="Reds_d", legend=False)
    plt.title("Model Comparison: MAE", fontsize=16)
    plt.ylabel("MAE")
    plt.xlabel("Model")
    plt.xticks(rotation=45)
    plt.grid(axis="y", alpha=0.75)
    plt.show()


def feature_importance_analysis(trained_model, feature_names):
    """
    Visualizes feature importance for a given trained model.

    Parameters:
        trained_model: A fitted model (e.g., RandomForest, GradientBoosting, XGBoost).
        feature_names (list): List of feature names corresponding to the model input.
    """
    # Extract feature importances
    feature_importances = trained_model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": feature_importances
    }).sort_values(by="Importance", ascending=False)

    # Plot feature importances with corrected hue parameter
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x="Importance", y="Feature", hue="Feature", dodge=False, palette="viridis", legend=False)
    plt.title("Feature Importance", fontsize=16)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.grid(axis="x", alpha=0.75)
    plt.show()


# Execute visualization steps
if __name__ == "__main__":
    # Visualize metrics
    visualize_metrics("model_comparison_metrics.csv")

    # Example: Pass the trained Random Forest model
    from sklearn.ensemble import RandomForestRegressor
    
    # Replace with your actual trained Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Load the RFM data for training
    rfm_df = pd.read_csv("rfm_metrics.csv")
    X = rfm_df[["Recency", "Frequency", "Monetary"]]
    y = rfm_df["Frequency"] * rfm_df["Monetary"]  # Example target (adjust based on actual CLV logic)
    model.fit(X, y)

    # Visualize feature importance
    feature_importance_analysis(model, feature_names=["Recency", "Frequency", "Monetary"])