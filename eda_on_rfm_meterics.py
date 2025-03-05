import pandas as pd
import matplotlib.pyplot as plt

# Load the RFM dataset
rfm_df = pd.read_csv("rfm_metrics.csv")

# Function to plot histograms for RFM metrics
def plot_rfm_distributions(rfm_df):
    """
    Plots the distribution of RFM metrics (Recency, Frequency, Monetary).

    Parameters:
        rfm_df (DataFrame): RFM metrics dataset.
    """
    metrics = ["Recency", "Frequency", "Monetary"]
    for metric in metrics:
        plt.figure(figsize=(8, 4))
        plt.hist(rfm_df[metric], bins=30, alpha=0.7, edgecolor="black")
        plt.title(f"Distribution of {metric}")
        plt.xlabel(metric)
        plt.ylabel("Frequency")
        plt.grid(axis="y", alpha=0.75)
        plt.show()

# Function to calculate correlation matrix for RFM metrics
def calculate_rfm_correlation(rfm_df):
    """
    Calculates and returns the correlation matrix of RFM metrics.

    Parameters:
        rfm_df (DataFrame): RFM metrics dataset.

    Returns:
        correlation_matrix (DataFrame): Correlation matrix of Recency, Frequency, and Monetary.
    """
    correlation_matrix = rfm_df[["Recency", "Frequency", "Monetary"]].corr()
    return correlation_matrix

# Plot distributions
plot_rfm_distributions(rfm_df)

# Calculate and print correlation matrix
correlation_matrix = calculate_rfm_correlation(rfm_df)
print("Correlation Matrix for RFM Metrics:")
print(correlation_matrix)

# Save correlation matrix to a CSV file for further analysis
correlation_matrix.to_csv("rfm_correlation_matrix.csv", index=True)
print("Correlation matrix saved as 'rfm_correlation_matrix.csv'.")