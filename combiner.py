import pandas as pd

# Paths to your two CSV files
csv1_path = "fraudtest.csv"
csv2_path = "fraudTrain.csv"

# Read both CSVs
df1 = pd.read_csv(csv1_path)
df2 = pd.read_csv(csv2_path)

# Combine them (stack rows)
combined_df = pd.concat([df1, df2], ignore_index=True)

# Save the result
combined_df.to_csv("transactions.csv", index=False)


