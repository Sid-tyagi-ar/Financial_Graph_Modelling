
import pandas as pd

df = pd.read_csv('data/raw/processed.csv')
print(df['time_bucket'].unique())
