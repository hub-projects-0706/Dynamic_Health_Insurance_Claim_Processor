import numpy as np
import pandas as pd
import os

os.makedirs('ML-Project/data', exist_ok=True)
np.random.seed(42)
n_samples = 500

customer_ids = [f'CUST-{1000 + i}' for i in range(n_samples)]
age = np.random.randint(18, 70, size=n_samples)
tenure_months = np.random.randint(1, 72, size=n_samples)
monthly_charges = np.round(np.random.uniform(20.0, 120.0, size=n_samples), 2)
total_charges = np.round(monthly_charges * tenure_months + np.random.normal(0, 10, size=n_samples), 2)
total_charges = np.maximum(total_charges, monthly_charges)
num_support_tickets = np.random.poisson(lam=1.5, size=n_samples)

contract_types = np.random.choice(['Month-to-Month', 'One-Year', 'Two-Year'], size=n_samples, p=[0.5, 0.3, 0.2])
paperless_billing = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.6, 0.4])
payment_methods = np.random.choice(['Electronic Check', 'Mailed Check', 'Bank Transfer', 'Credit Card'], size=n_samples)

log_odds = (
    -1.5 
    + 0.03 * (age - 40)
    - 0.05 * tenure_months
    + 0.025 * (monthly_charges - 60)
    + 0.8 * num_support_tickets
    + (contract_types == 'Month-to-Month') * 1.2
    - (contract_types == 'Two-Year') * 1.0
)
prob = 1 / (1 + np.exp(-log_odds))
churn = (np.random.rand(n_samples) < prob).astype(int)

df = pd.DataFrame({
    'customer_id': customer_ids,
    'age': age,
    'tenure_months': tenure_months,
    'monthly_charges': monthly_charges,
    'total_charges': total_charges,
    'num_support_tickets': num_support_tickets,
    'contract_type': contract_types,
    'paperless_billing': paperless_billing,
    'payment_method': payment_methods,
    'churn': churn
})

df.to_csv('ML-Project/data/dataset.csv', index=False)
churn_rate = df['churn'].mean()
print(f"Successfully generated dataset.csv with {len(df)} rows and target churn rate of {churn_rate:.2%}")
