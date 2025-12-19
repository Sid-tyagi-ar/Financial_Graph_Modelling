import pandas as pd
import numpy as np
from haversine import haversine_vector, Unit
from tqdm import tqdm

def analyze_transaction_distances(filepath="data/preprocessed/processed.csv"):
    """
    Analyzes the geographical distance between sequential transactions for each user.

    Args:
        filepath (str): Path to the preprocessed CSV file.
    """
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        print("Please ensure the preprocessing script has been run successfully.")
        return

    required_cols = ['cc_num', 'unix_time', 'merch_lat', 'merch_long']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: The CSV must contain the following columns: {required_cols}")
        return

    print("Calculating distances between sequential transactions for each user...")
    
    # Sort by user and time to prepare for sequential calculation
    df = df.sort_values(by=['cc_num', 'unix_time']).reset_index(drop=True)

    # Group by user
    grouped = df.groupby('cc_num')

    all_distances = []

    # Use tqdm for progress bar
    for _, group in tqdm(grouped, desc="Processing users"):
        if len(group) > 1:
            # Get coordinates of current and next transaction
            coords1 = group[['merch_lat', 'merch_long']][:-1].values
            coords2 = group[['merch_lat', 'merch_long']][1:].values
            
            # Calculate haversine distance for all sequential pairs in the group
            distances = haversine_vector(coords1, coords2, Unit.KILOMETERS)
            all_distances.extend(distances)

    if not all_distances:
        print("No sequential transactions found to analyze.")
        return

    # Convert to numpy array for statistical analysis
    distances_arr = np.array(all_distances)

    # Remove potential NaNs
    distances_arr = distances_arr[~np.isnan(distances_arr)]

    print("\n--- Geographical Distance Analysis (in Kilometers) ---")
    
    # Define percentiles to calculate
    percentiles = [10, 25, 50, 75, 90, 95, 99, 99.5, 99.9]
    p_values = np.percentile(distances_arr, percentiles)

    print(f"Total sequential transaction pairs analyzed: {len(distances_arr)}")
    print(f"Mean distance: {np.mean(distances_arr):.2f} km")
    print(f"Median distance: {np.median(distances_arr):.2f} km")
    print(f"Standard Deviation: {np.std(distances_arr):.2f} km")
    print(f"Min distance: {np.min(distances_arr):.2f} km")
    print(f"Max distance: {np.max(distances_arr):.2f} km")
    
    print("\nPercentile Distribution:")
    for p, val in zip(percentiles, p_values):
        print(f"  {p}th percentile: {val:.2f} km")
    
    print("\nThis report shows the distribution of distances between consecutive transactions.")
    print("Use these percentiles to choose a data-driven threshold for defining a 'close-by' transaction edge.")

if __name__ == "__main__":
    analyze_transaction_distances()
