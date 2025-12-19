
import torch
import os
from glob import glob
from tqdm import tqdm

def find_same_merchant_edge():
    """
    Iterates through all graphs in the training set to find one that has an
    edge connecting two transaction nodes with the same merchant ID.
    """
    print("Searching for a graph with a same-merchant transaction edge...")
    
    # Path to the preprocessed training graphs
    graph_files = sorted(glob(os.path.join('.', 'processed_graphs', 'train', '*.pt')))

    if not graph_files:
        print("Error: No preprocessed training graph files found.")
        return

    found_graph_index = -1

    for i, file_path in enumerate(tqdm(graph_files, desc="Scanning graphs")):
        # Load the raw graph data. This contains integer indices, not one-hot encoded features.
        raw_data = torch.load(file_path, weights_only=False)

        # Ensure the graph has nodes and edges to check
        if raw_data.x is None or raw_data.edge_index is None or raw_data.x.shape[0] < 2:
            continue

        node_features = raw_data.x
        edge_index = raw_data.edge_index

        # Iterate through each edge
        for j in range(edge_index.shape[1]):
            u, v = edge_index[0, j].item(), edge_index[1, j].item()

            # We are looking for edges between two transaction nodes.
            # Node 0 is the root (person) node, so transaction nodes have index > 0.
            if u > 0 and v > 0:
                # From preprocess_transactions.py, the merchant ID is the 2nd feature (index 1)
                # in the feature vector for transaction nodes.
                merchant_u = node_features[u, 1]
                merchant_v = node_features[v, 1]

                if merchant_u == merchant_v:
                    found_graph_index = i
                    break  # Exit inner loop once found
        
        if found_graph_index != -1:
            break # Exit outer loop once found

    if found_graph_index != -1:
        print(f"\nSuccess! Found a graph with a same-merchant edge.")
        print(f"Graph index: {found_graph_index}")
        print(f"File: {graph_files[found_graph_index]}")
    else:
        print("\nSearch complete. No graph found with an edge connecting two transactions from the same merchant.")

if __name__ == '__main__':
    find_same_merchant_edge()
