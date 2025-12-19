
import os
import pickle
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import numpy as np

# Assuming NO_VALUE_FOR_FEATURE_IDX is 0, as defined in preprocess_transactions.py
NO_VALUE_FOR_FEATURE_IDX = 0

def load_graph_and_metadata(graph_path: str, encoders_path: str, bin_info_path: str):
    """
    Loads a PyG graph, encoders, and bin_info.
    """
    graph = torch.load(graph_path, weights_only=False)
    with open(encoders_path, 'rb') as f:
        encoders = pickle.load(f)
    with open(bin_info_path, 'rb') as f:
        bin_info = pickle.load(f)
    return graph, encoders, bin_info

def decode_node_features(x: torch.Tensor, encoders: dict, bin_info: dict):
    """
    Decodes one-hot encoded node features.
    x: (num_nodes, total_node_features_dim) tensor
    """
    decoded_nodes = []
    
    # Node type: 0 for transaction, 1 for root
    # In preprocess_transactions.py, node_type_one_hot is 2 dims, then other_features are concatenated
    # So, first 2 dims are node type
    node_type_ohe = x[:, :2]
    node_type_idx = torch.argmax(node_type_ohe, dim=1)
    
    # Other node features start from index 2
    current_dim_idx = 2
    
    # Amount
    num_amt_bins = bin_info['amt'].shape[0] - 1
    amt_ohe = x[:, current_dim_idx : current_dim_idx + num_amt_bins]
    amt_bin_idx = torch.argmax(amt_ohe, dim=1)
    amt_decoded = [bin_info['amt'][idx] if idx != NO_VALUE_FOR_FEATURE_IDX else 'N/A' for idx in amt_bin_idx]
    current_dim_idx += num_amt_bins

    # Merchant
    num_merch_classes = len(encoders['merchant'].classes_)
    merch_ohe = x[:, current_dim_idx : current_dim_idx + num_merch_classes]
    merch_idx = torch.argmax(merch_ohe, dim=1)
    merch_decoded = [encoders['merchant'].inverse_transform([idx.item()])[0] for idx in merch_idx]
    current_dim_idx += num_merch_classes

    # Category
    num_cat_classes = len(encoders['category'].classes_)
    cat_ohe = x[:, current_dim_idx : current_dim_idx + num_cat_classes]
    cat_idx = torch.argmax(cat_ohe, dim=1)
    cat_decoded = [encoders['category'].inverse_transform([idx.item()])[0] for idx in cat_idx]
    current_dim_idx += num_cat_classes

    for i in range(x.size(0)):
        node_type = 'root' if node_type_idx[i] == 1 else 'transaction'
        node_info = {
            'node_id': i,
            'type': node_type
        }
        if node_type == 'transaction':
            node_info['amount_bin'] = amt_decoded[i]
            node_info['merchant'] = merch_decoded[i]
            node_info['category'] = cat_decoded[i]
        else: # Root node
            node_info['amount_bin'] = 'N/A'
            node_info['merchant'] = 'N/A'
            node_info['category'] = 'N/A'
        decoded_nodes.append(node_info)
    
    return decoded_nodes

def decode_edge_features(edge_attr: torch.Tensor, edge_index: torch.Tensor, bin_info: dict):
    """
    Decodes one-hot encoded edge features.
    edge_attr: (num_edges, total_edge_features_dim) tensor
    edge_index: (2, num_edges) tensor
    """
    decoded_edges = []
    
    if edge_attr.numel() == 0:
        return []

    current_dim_idx = 0

    # is_root_txn_edge (2 classes)
    is_root_txn_ohe = edge_attr[:, current_dim_idx : current_dim_idx + 2]
    is_root_txn_idx = torch.argmax(is_root_txn_ohe, dim=1)
    current_dim_idx += 2

    # is_txn_txn_edge (2 classes)
    is_txn_txn_ohe = edge_attr[:, current_dim_idx : current_dim_idx + 2]
    is_txn_txn_idx = torch.argmax(is_txn_txn_ohe, dim=1)
    current_dim_idx += 2

    # city_pop_bin
    num_city_pop_bins = bin_info['city_pop_total_bins']
    city_pop_ohe = edge_attr[:, current_dim_idx : current_dim_idx + num_city_pop_bins]
    city_pop_bin_idx = torch.argmax(city_pop_ohe, dim=1)
    city_pop_decoded = [bin_info['city_pop'][idx - 1] if idx != NO_VALUE_FOR_FEATURE_IDX else 'N/A' for idx in city_pop_bin_idx]
    current_dim_idx += num_city_pop_bins

    # merch_lat_bin
    num_merch_lat_bins = bin_info['merch_lat_total_bins']
    merch_lat_ohe = edge_attr[:, current_dim_idx : current_dim_idx + num_merch_lat_bins]
    merch_lat_bin_idx = torch.argmax(merch_lat_ohe, dim=1)
    merch_lat_decoded = [bin_info['merch_lat'][idx - 1] if idx != NO_VALUE_FOR_FEATURE_IDX else 'N/A' for idx in merch_lat_bin_idx]
    current_dim_idx += num_merch_lat_bins

    # merch_lon_bin
    num_merch_lon_bins = bin_info['merch_lon_total_bins']
    merch_lon_ohe = edge_attr[:, current_dim_idx : current_dim_idx + num_merch_lon_bins]
    merch_lon_bin_idx = torch.argmax(merch_lon_ohe, dim=1)
    merch_lon_decoded = [bin_info['merch_lon'][idx - 1] if idx != NO_VALUE_FOR_FEATURE_IDX else 'N/A' for idx in merch_lon_bin_idx]
    current_dim_idx += num_merch_lon_bins

    # unix_time_bin
    num_unix_time_bins = bin_info['unix_time_total_bins']
    unix_time_ohe = edge_attr[:, current_dim_idx : current_dim_idx + num_unix_time_bins]
    unix_time_bin_idx = torch.argmax(unix_time_ohe, dim=1)
    unix_time_decoded = [bin_info['unix_time'][idx - 1] if idx != NO_VALUE_FOR_FEATURE_IDX else 'N/A' for idx in unix_time_bin_idx]
    current_dim_idx += num_unix_time_bins

    # shared_merchant_category (2 classes)
    shared_merch_cat_ohe = edge_attr[:, current_dim_idx : current_dim_idx + 2]
    shared_merch_cat_idx = torch.argmax(shared_merch_cat_ohe, dim=1)
    current_dim_idx += 2

    for i in range(edge_attr.size(0)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        
        # Only decode if it's an actual edge (not all NO_VALUE_FOR_FEATURE_IDX)
        if is_root_txn_idx[i] == 0 and is_txn_txn_idx[i] == 0:
            continue # This is a 'no edge' representation

        edge_info = {
            'source': src,
            'destination': dst,
            'is_root_transaction_edge': bool(is_root_txn_idx[i]),
            'is_transaction_transaction_edge': bool(is_txn_txn_idx[i]),
            'city_pop_bin': city_pop_decoded[i],
            'merch_lat_bin': merch_lat_decoded[i],
            'merch_lon_bin': merch_lon_decoded[i],
            'unix_time_bin': unix_time_decoded[i],
            'shared_merchant_category': bool(shared_merch_cat_idx[i])
        }
        decoded_edges.append(edge_info)
    
    return decoded_edges


def main():
    # Example usage: replace with actual paths
    project_root = os.getcwd()
    graph_path = os.path.join(project_root, 'processed', 'train_processed.pt') # Load processed graph
    encoders_path = os.path.join(project_root, 'encoders.pkl')
    bin_info_path = os.path.join(project_root, 'bin_info.pkl')

    if not os.path.exists(graph_path):
        print(f"Error: Graph file not found at {graph_path}")
        print("Please ensure you have run preprocess_transactions.py to generate processed graphs.")
        return
    if not os.path.exists(encoders_path):
        print(f"Error: Encoders file not found at {encoders_path}")
        print("Please ensure you have run preprocess_transactions.py to generate encoders.pkl.")
        return
    if not os.path.exists(bin_info_path):
        print(f"Error: Bin info file not found at {bin_info_path}")
        print("Please ensure you have run preprocess_transactions.py to generate bin_info.pkl.")
        return

    loaded_graph_batch, encoders, bin_info = load_graph_and_metadata(graph_path, encoders_path, bin_info_path)
    # loaded_graph_batch is a single Data object representing a batch of graphs
    # Extract the first graph from the batch
    graph = loaded_graph_batch.to_data_list()[0]

    print(f"--- Decoding Graph: {graph_path} ---")
    print(f"Number of nodes: {graph.num_nodes}")
    print(f"Number of edges: {graph.edge_index.size(1) // 2} (undirected)")
    print(f"Graph-level label (y): {graph.y}")

    decoded_nodes = decode_node_features(graph.x, encoders, bin_info)
    print("\n--- Decoded Node Features ---")
    for node in decoded_nodes:
        print(node)

    decoded_edges = decode_edge_features(graph.edge_attr, graph.edge_index, bin_info)
    print("\n--- Decoded Edge Features ---")
    # Filter out duplicate undirected edges for display
    seen_edges = set()
    for edge in decoded_edges:
        u, v = edge['source'], edge['destination']
        if (u, v) not in seen_edges and (v, u) not in seen_edges:
            print(edge)
            seen_edges.add((u, v))

if __name__ == '__main__':
    main()
