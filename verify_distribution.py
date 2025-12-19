
import torch
import torch.nn.functional as F
import pickle
from glob import glob

def get_feature_dims():
    """ Load metadata to get feature dimensions. """
    try:
        with open('bin_info.pkl', 'rb') as f:
            bin_info = pickle.load(f)
        return {
            'is_root_txn_edge': 2,
            'is_txn_txn_edge': 2,
            'city_pop': bin_info['city_pop_total_bins'],
            'merch_lat': bin_info['merch_lat_total_bins'],
            'merch_lon': bin_info['merch_lon_total_bins'],
            'unix_time': bin_info['unix_time_total_bins'],
            'shared_merchant_category': 2
        }
    except FileNotFoundError:
        print("Error: bin_info.pkl not found. Make sure you are in the project root.")
        return None

def verify_distribution():
    """
    Loads one graph file and analyzes its edge distribution in two ways:
    1. From the raw integer data (ground truth).
    2. From the one-hot encoded data (as used by the model).
    """
    print("--- Running Final Verification Script ---")
    
    # --- Load Data ---
    graph_files = sorted(glob('processed_graphs/train/*.pt'))
    if not graph_files:
        print("Error: No graph files found.")
        return
        
    graph_file = graph_files[0]
    print(f"Analyzing file: {graph_file}")
    raw_data = torch.load(graph_file, weights_only=False)
    edge_attrs_int = raw_data.edge_attr.long()  # The integer features, shape (num_edges, 7)

    # --- Method A: Ground Truth from Integers ---
    print("\n--- Method A: Analyzing Raw Integer Data ---")
    is_root_edge = edge_attrs_int[:, 0] == 1
    is_txn_edge = edge_attrs_int[:, 1] == 1
    is_no_edge = (~is_root_edge) & (~is_txn_edge)
    
    count_root = torch.sum(is_root_edge).item()
    count_txn = torch.sum(is_txn_edge).item()
    count_none = torch.sum(is_no_edge).item()
    total_a = count_root + count_txn + count_none

    if total_a == 0:
        print("No edges found in file to analyze.")
        return

    print(f"  Ground Truth - P(is_root_txn=1):    {count_root / total_a:.4f}")
    print(f"  Ground Truth - P(is_txn_txn=1):     {count_txn / total_a:.4f}")
    print(f"  Ground Truth - P(No Edge Flag):    {count_none / total_a:.4f}")
    print("-" * 50)

    # --- Method B: My Logic on One-Hot Data ---
    print("\n--- Method B: Analyzing One-Hot Encoded Data ---")
    num_edge_features = get_feature_dims()
    if num_edge_features is None:
        return

    # Replicate the one-hot encoding from TransactionDataset.get()
    ohe_features = []
    for i, (key, num_cats) in enumerate(num_edge_features.items()):
        ohe_features.append(F.one_hot(edge_attrs_int[:, i], num_classes=num_cats))
    
    edge_attrs_ohe = torch.cat(ohe_features, dim=1).float()

    # Replicate the logic from the simple edge_counts()
    counts_b = edge_attrs_ohe.sum(dim=0)
    
    # To get the probability of a feature, we need to normalize within its own block
    # The first feature (is_root_txn_edge) has 2 classes
    prob_is_root_1_calc = counts_b[1] / (counts_b[0] + counts_b[1])

    # The second feature (is_txn_txn_edge) has 2 classes and starts at index 2
    prob_is_txn_1_calc = counts_b[3] / (counts_b[2] + counts_b[3])

    print(f"  Calculated - P(is_root_txn=1):    {prob_is_root_1_calc:.4f}")
    print(f"  Calculated - P(is_txn_txn=1):     {prob_is_txn_1_calc:.4f}")
    print("-" * 50)

    print("\nConclusion: Compare the probabilities from Method A and Method B.")
    print("If they match, the one-hot encoding and summation logic is correct.")
    print("If they do not match, there is a hidden issue in the data loading or encoding.")

if __name__ == '__main__':
    verify_distribution()
