
import pandas as pd
import pickle

with open('data/encoders/encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

df = pd.read_csv('data/preprocessed/processed.csv')

print("NaN values in y features:")
print("age_bucket:", df['age_bucket'].isnull().sum())
