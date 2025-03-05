import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

# Load data
@st.cache_data
def load_data():
    rfm_metrics = pd.read_csv("rfm_metrics.csv")
    rfm_with_predictions = pd.read_csv("rfm_with_predictions.csv")
    model_metrics = pd.read_csv("model_comparison_metrics.csv")
    return rfm_metrics, rfm_with_predictions, model_metrics

# Segment customers based on predicted CLV
def segment_customers(df):
    quantiles = df["Predicted_CLV"].quantile([0.33, 0.67]).values
    conditions = [
        df["Predicted_CLV"] <= quantiles[0],
        (df["Predicted_CLV"] > quantiles[0]) & (df["Predicted_CLV"] <= quantiles[1]),
        df["Predicted_CLV"] > quantiles[1]
    ]
    choices = ["Low CLV", "Medium CLV", "High CLV"]
    df["Segment"] = pd.cut(df["Predicted_CLV"], bins=[-float('inf'), quantiles[0], quantiles[1], float('inf')], labels=choices)
    return df

# Main dashboard
def main():
    # Load data
    rfm_metrics, rfm_with_predictions, model_metrics = load_data()

    # Title
    st.title("Customer Lifetime Value (CLV) Dashboard")
    st.write("""
        This dashboard provides insights into customer lifetime value predictions, RFM metrics, 
        and model performance. Use the visualizations to analyze customer segments and gain actionable insights.
    """)

    # Customer Segments
    st.header("1. Customer Segments Overview")
    rfm_with_predictions = segment_customers(rfm_with_predictions)
    segment_counts = rfm_with_predictions["Segment"].value_counts()
    st.bar_chart(segment_counts)
    st.write("""
        **Explanation:** The bar chart shows the distribution of customers across Low, Medium, and High CLV segments. 
        This helps identify which group contributes the most to the overall lifetime value.
    """)

    # Feature Importance
    st.header("2. Feature Importance")
    st.write("Below is the feature importance for predicting CLV, based on the trained model.")
    feature_importance_image = Image.open("Figure_7.png")
    st.image(feature_importance_image, caption="Feature Importance for CLV Prediction")

    # Model Metrics
    st.header("3. Model Performance Comparison")
    st.write("The table below shows the performance of different models used for predicting CLV.")
    st.dataframe(model_metrics)
    st.write("""
        **Explanation:** Use this table to compare model performance based on R² Score, RMSE, and MAE. 
        Lower RMSE and MAE indicate better predictions.
    """)

    # RFM Metrics
    st.header("4. RFM Insights")
    st.write("Summary statistics of RFM metrics for each customer segment.")
    rfm_summary = rfm_with_predictions.groupby("Segment")[["Recency", "Frequency", "Monetary", "Predicted_CLV"]].mean()
    st.dataframe(rfm_summary)
    st.write("""
        **Explanation:** This table summarizes the average Recency, Frequency, Monetary, and Predicted CLV for each segment.
        Use it to understand customer behavior within each group.
    """)

    # Individual Customer Drill-Down
    st.header("5. Customer Details")
    customer_id = st.selectbox("Select Customer ID", rfm_with_predictions["CustomerID"].unique())
    customer_data = rfm_with_predictions[rfm_with_predictions["CustomerID"] == customer_id]
    st.write("**Customer Information:**")
    st.dataframe(customer_data)

    # Download Data
    st.header("6. Download Data")
    st.download_button(
        label="Download Model Metrics CSV",
        data=model_metrics.to_csv(index=False).encode('utf-8'),
        file_name="model_metrics.csv",
        mime="text/csv"
    )
    st.download_button(
        label="Download RFM Metrics CSV",
        data=rfm_metrics.to_csv(index=False).encode('utf-8'),
        file_name="rfm_metrics.csv",
        mime="text/csv"
    )
    st.download_button(
        label="Download Predicted CLV CSV",
        data=rfm_with_predictions.to_csv(index=False).encode('utf-8'),
        file_name="rfm_with_predictions.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()