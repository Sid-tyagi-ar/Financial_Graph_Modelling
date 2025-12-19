import pandas as pd

CSV_PATH = "data/preprocessed/processed.csv"
df = pd.read_csv(CSV_PATH)

# Must match preprocess_transactions_improved.py EXACTLY
X_cat = [
    "merchant_cluster",
    "category",           # changed name
    "user_city",
    "user_job",
    "time_bucket_hours",
]

E_cat = [
    "amt_bin",
    "edge_time_delta_bucket",
]

y_cat = [
    "gender",
    "city",
    "job",
    "age_bucket",
]

def analyze_feature(df, col, topk=10):
    print(f"\n▶ Feature: {col}")
    if col not in df.columns:
        print(f"  ❌ Column NOT found in CSV!")
        return
    vc = df[col].value_counts(dropna=False)
    print(f"  #classes: {len(vc)}")
    print(f"  rare classes (<=5): {(vc <= 5).sum()}")
    print(f"  top {topk}:")
    print(vc.head(topk))


print("\n==================== NODE FEATURES ====================")
for col in X_cat:
    analyze_feature(df, col)

print("\n==================== EDGE FEATURES ====================")
for col in E_cat:
    analyze_feature(df, col)

print("\n==================== Y FEATURES ====================")
for col in y_cat:
    analyze_feature(df, col)
