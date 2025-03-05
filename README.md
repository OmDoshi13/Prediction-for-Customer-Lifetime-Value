# Customer Lifetime Value (CLV) Prediction

## Overview

This project focuses on predicting **Customer Lifetime Value (CLV)** by leveraging **RFM metrics** (Recency, Frequency, and Monetary) along with machine learning models. It also features an **interactive Streamlit dashboard** for exploring customer insights and an **API** powered by FastAPI for real-time predictions.

By utilizing this tool, businesses can:
- Identify their most valuable customers,
- Predict customer churn,
- Optimize marketing strategies for better engagement and profitability.

---

## Features

1. **CLV Prediction:**
   - Estimate customer lifetime value using RFM-based machine learning models.

2. **Interactive Dashboard:**
   - Utilize Streamlit to analyze customer segmentation (High, Medium, Low CLV) and key RFM metrics.

3. **Real-Time Prediction API:**
   - FastAPI enables seamless integration for real-time CLV forecasting.

4. **Advanced Customer Analytics:**
   - Includes customer churn prediction and segmentation through clustering techniques.

5. **Comprehensive Visualizations:**
   - Compare different model performances, examine RFM metric distributions, and evaluate feature importance.

---

## Project Structure

```bash
CLV_Prediction/
├── clv_api.py                  # FastAPI setup for real-time CLV predictions
├── enhanced_dashboard.py       # Streamlit-based visualization dashboard
├── trained_clv_model.pkl       # Pre-trained Random Forest model
├── rfm_metrics.csv             # Dataset containing RFM metric values
├── rfm_with_predictions.csv    # Dataset with predicted CLV scores
├── model_comparison_metrics.csv # Performance metrics of different models
├── Figures/                    # Folder containing generated visualizations
│   ├── Figure_1.png
│   ├── Figure_2.png
│   └── Figure_7.png
├── README.md                   # Project documentation
├── requirements.txt            # List of dependencies required to run the project
```

---

## Key Visualizations

1. **RFM Distributions:**
   - Examine how Recency, Frequency, and Monetary metrics are distributed among customers.

2. **Customer Segmentation:**
   - Visualize High, Medium, and Low CLV customers for strategic decision-making.

3. **Feature Importance Analysis:**
   - Understand how different RFM metrics influence CLV prediction.

---

## Technology Stack

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **Visualization Tools:** Matplotlib, Seaborn, Streamlit
- **API Development:** FastAPI
- **Analytical Techniques:** K-Means Clustering, Logistic Regression for churn analysis

---

## Use Cases

1. **Customer Segmentation:**
   - Recognize and retain high-value customers.
   - Target low-value customers with personalized marketing campaigns.

2. **Churn Prediction:**
   - Analyze customer behavior to proactively reduce churn rates.

3. **Optimizing Business Strategies:**
   - Improve budget allocation and refine customer engagement based on CLV insights.

---

## License

This project is distributed under the **MIT License**.
