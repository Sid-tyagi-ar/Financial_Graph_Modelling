
import pandas as pd
import pickle

with open('data/encoders/encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

df = pd.read_csv('data/preprocessed/processed.csv')

print("Max values in csv:")
print("gender_id max:", df['gender_id'].max())
print("city_id max:", df['city_id'].max())
print("job_id max:", df['job_id'].max())
print("age_bucket max:", df['age_bucket'].max())

print("\nCardinalities from encoders:")
print("gender:", len(encoders['gender']))
print("city:", len(encoders['city']))
print("job:", len(encoders['job']))
print("age_bucket:", len(encoders['age_bucket']))
