import pandas as pd
import random
from faker import Faker

def generate_synthetic_data(num_customers=7000, num_transactions=210000):
    """
    Generates synthetic customer profiles and transaction data, and combines them into one dataset.
    
    Parameters:
        num_customers (int): Number of unique customers to generate.
        num_transactions (int): Number of transactions to generate.
    
    Returns:
        combined_df (DataFrame): Combined customer and transaction data.
    """
    # Initialize Faker for generating realistic data
    fake = Faker()
    PRODUCT_CATEGORIES = ["Electronics", "Clothing", "Home & Kitchen", "Books", "Sports"]
    
    # Generate synthetic customer profiles
    customer_data = []
    for customer_id in range(1, num_customers + 1):
        age = random.randint(18, 65)  # Random age between 18 and 65
        gender = random.choice(["Male", "Female"])
        region = random.choice(["North", "South", "East", "West"])
        customer_data.append([customer_id, age, gender, region])

    customer_df = pd.DataFrame(customer_data, columns=["CustomerID", "Age", "Gender", "Region"])

    # Generate synthetic transaction data
    transaction_data = []
    for _ in range(num_transactions):
        customer_id = random.randint(1, num_customers)
        transaction_id = fake.uuid4()
        transaction_date = fake.date_between(start_date="-1y", end_date="today")
        transaction_amount = round(random.uniform(10, 5000), 2)  # Random amount between $10 and $500
        product_category = random.choice(PRODUCT_CATEGORIES)
        transaction_data.append([transaction_id, customer_id, transaction_date, transaction_amount, product_category])

    transaction_df = pd.DataFrame(transaction_data, columns=[
        "TransactionID", "CustomerID", "TransactionDate", "TransactionAmount", "ProductCategory"
    ])

    # Combine the datasets based on CustomerID
    combined_df = pd.merge(transaction_df, customer_df, on="CustomerID", how="left")

    # Save datasets to CSV
    customer_df.to_csv("customer_profiles.csv", index=False)
    transaction_df.to_csv("transaction_data.csv", index=False)
    combined_df.to_csv("combined_data.csv", index=False)

    return combined_df

# Generate and save data with default parameters
if __name__ == "__main__":
    combined_data = generate_synthetic_data(num_customers=7000, num_transactions=210000)
    print("Datasets created and saved as 'customer_profiles.csv', 'transaction_data.csv', and 'combined_data.csv'.")