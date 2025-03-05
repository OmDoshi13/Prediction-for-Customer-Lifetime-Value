# Customer Lifetime Value (CLV) Prediction

## Overview

This project predicts **Customer Lifetime Value (CLV)** using **RFM metrics** (Recency, Frequency, and Monetary) and machine learning models. It includes an **interactive Streamlit dashboard** for visualizing customer insights and an **API** built with FastAPI for real-time CLV predictions. 

This tool helps businesses:
- Identify high-value customers,
- Predict churn,
- Optimize marketing strategies.

---

## Features

1. **CLV Prediction:**
   - Predict customer lifetime value based on RFM metrics using trained machine learning models.

2. **Streamlit Dashboard:**
   - Visualize customer segments (High, Medium, Low CLV) and key metrics like RFM distributions and feature importance.

3. **Real-Time API:**
   - Use FastAPI for real-time CLV predictions, enabling seamless integration with external systems.

4. **Advanced Analytics:**
   - Includes churn prediction and customer clustering for deeper insights.

5. **Visualizations:**
   - Compare model performance, explore RFM distributions, and understand feature importance.

---

## Project Structure

```bash
CLV_Prediction/
├── clv_api.py                  # FastAPI implementation for real-time predictions
├── enhanced_dashboard.py       # Streamlit dashboard for visualization
├── trained_clv_model.pkl       # Trained Random Forest model
├── rfm_metrics.csv             # RFM metrics dataset
├── rfm_with_predictions.csv    # Dataset with predicted CLV values
├── model_comparison_metrics.csv # Model performance metrics
├── Figures/                    # Folder for visualizations
│   ├── Figure_1.png
│   ├── Figure_2.png
│   └── Figure_7.png
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
```

---

## Key Visualizations

1. **RFM Distributions:**
   - Understand the distribution of Recency, Frequency, and Monetary values.

2. **Customer Segments:**
   - Visualize High, Medium, and Low CLV customers.

3. **Feature Importance:**
   - Explore the contribution of RFM metrics to CLV prediction.

---

## Technology Stack

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **Visualization:** Matplotlib, Seaborn, Streamlit
- **API Development:** FastAPI
- **Clustering and Churn Analytics:** K-Means, Logistic Regression

---

## Use Cases

1. **Customer Segmentation:**
   - Identify and retain high-value customers.
   - Re-engage low-value customers with targeted campaigns.

2. **Churn Prediction:**
   - Predict and prevent customer churn using RFM insights.

3. **Business Strategy Optimization:**
   - Allocate marketing budgets and optimize customer engagement based on CLV predictions.

---

## License

This project is licensed under the **MIT License**.
