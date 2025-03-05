import pandas as pd

def calculate_rfm(combined_data):
    """
    Calculates RFM metrics from combined customer and transaction data.

    Parameters:
        combined_data (DataFrame): Combined dataset with customer and transaction details.

    Returns:
        rfm_df (DataFrame): Customer-level dataset with RFM metrics.
    """
    # Ensure TransactionDate is in datetime format
    combined_data["TransactionDate"] = pd.to_datetime(combined_data["TransactionDate"])

    # Calculate snapshot date (latest date in the dataset)
    snapshot_date = combined_data["TransactionDate"].max()

    # Calculate RFM metrics
    rfm_df = combined_data.groupby("CustomerID").agg(
        Recency=("TransactionDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("TransactionID", "count"),
        Monetary=("TransactionAmount", "sum"),
    ).reset_index()

    return rfm_df


# Load the combined data (ensure the path is correct)
combined_data = pd.read_csv("combined_data.csv")

# Calculate RFM metrics
rfm_df = calculate_rfm(combined_data)

# Save the RFM metrics to a CSV file
rfm_df.to_csv("rfm_metrics.csv", index=False)
print("RFM metrics saved to 'rfm_metrics.csv'.")