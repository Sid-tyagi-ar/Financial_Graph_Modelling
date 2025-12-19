
import pandas as pd
import pickle

with open('data/encoders/encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

df = pd.read_csv('data/preprocessed/processed.csv')

print("Min values in csv:")
print("gender_id min:", df['gender_id'].min())
print("city_id min:", df['city_id'].min())
print("job_id min:", df['job_id'].min())
print("age_bucket min:", df['age_bucket'].min())
