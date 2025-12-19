
import pandas as pd

df = pd.read_csv('transactions.csv', nrows=5)
print(df.columns)
