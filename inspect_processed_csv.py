
import pandas as pd

df = pd.read_csv('data/preprocessed/processed.csv')

columns_to_check = [
    'amt_bin',
    'merchant_cluster',
    'category_small_id',
    'geocell_id',
    'time_bucket',
    'gender_id',
    'city_id',
    'job_id',
    'age_bucket'
]

for col in columns_to_check:
    print(f"Column: {col}")
    print(f"  min: {df[col].min()}")
    print(f"  max: {df[col].max()}")
    print(f"  unique values: {df[col].nunique()}")
    print(f"  has NaNs: {df[col].isnull().any()}")
    print("-" * 20)
