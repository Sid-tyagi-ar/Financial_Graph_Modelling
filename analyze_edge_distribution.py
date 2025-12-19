import torch
import glob
from collections import Counter
import pandas as pd
import os

def analyze_edge_distribution(processed_dir="processed/train"):
    """
    Analyzes the distribution of edge features in a preprocessed graph dataset.

    Args:
        processed_dir (str): Path to the directory containing processed .pt graph files.
    """
    graph_files = glob.glob(os.path.join(processed_dir, "*.pt"))

    if not graph_files:
        print(f"Error: No graph files found in '{processed_dir}'.")
        print("Please ensure you have run the preprocessing script and specified the correct directory.")
        return

    # Default to old pipeline feature names
    feature_names = [
        'is_sequential', 
        'is_shared_attribute', 
        'time_delta_bin', 
        'geo_distance_bin', 
        'amount_delta_bin'
    ]
    
    # Try to detect pipeline version by inspecting encoders.pkl
    new_encoder_path = os.path.join('data', 'encoders', 'encoders.pkl')
    if os.path.exists(new_encoder_path):
        try:
            with open(new_encoder_path, 'rb') as f:
                import pickle
                encoders = pickle.load(f)
            if 'merchant_id_map' in encoders:
                print("Detected new pipeline via encoders.pkl.")
                feature_names = [
                    'is_sequential',
                    'same_merchant_cluster',
                    'same_user',
                    'time_delta_bin',
                    'is_close_geo'
                ]
        except Exception:
            print("Could not read new encoder file, assuming old pipeline.")


    num_features = len(feature_names)
    counters = [Counter() for _ in range(num_features)]
    total_edges = 0

    print(f"Analyzing {len(graph_files)} graph files...")

    for graph_file in graph_files:
        try:
            data = torch.load(graph_file, weights_only=False)
            if data.edge_attr is not None and data.edge_attr.numel() > 0:
                total_edges += data.edge_attr.shape[0]
                # Transpose to iterate over features
                for i, feature_column in enumerate(data.edge_attr.t()):
                    if i < num_features:
                        counters[i].update(feature_column.numpy())
        except Exception as e:
            print(f"Warning: Could not load or process {graph_file}. Error: {e}")

    print(f"\n--- Edge Feature Distribution Analysis ---")
    print(f"Found {total_edges} total edges across {len(graph_files)} graphs.\n")

    for i, name in enumerate(feature_names):
        print(f"Feature: '{name}'")
        counter = counters[i]
        total_count = sum(counter.values())
        if total_count == 0:
            print("  No data found for this feature.\n")
            continue

        # Sort by class index for consistent output
        sorted_items = sorted(counter.items())

        df = pd.DataFrame(sorted_items, columns=['Class', 'Count'])
        df['Percentage'] = (df['Count'] / total_count * 100).round(2).astype(str) + '%' 
        
        print(df.to_string(index=False))
        print("-" * 30)

if __name__ == "__main__":
    # Assuming the script is run from the root of the project directory
    analyze_edge_distribution()