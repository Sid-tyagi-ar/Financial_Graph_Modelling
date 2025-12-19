import torch
from collections import Counter
import os
from glob import glob
import pandas as pd
from torch_geometric.data import Data

def analyze_edge_distribution():
    """
    Loads preprocessed training graphs and computes the distribution of
    different edge types based on the new preprocessing logic.
    """
    print("Loading training graphs to analyze edge distribution...")
    
    train_dir = 'processed_graphs/train'
    if not os.path.exists(train_dir):
        print(f"Directory not found: {train_dir}")
        print("Have you run preprocess_transactions.py?")
        return

    graph_files = glob(os.path.join(train_dir, '*.pt'))
    if not graph_files:
        print("No graphs found in the training directory.")
        return

    edge_type_counts = Counter()
    time_delta_bins = Counter()
    geo_dist_bins = Counter()
    amt_delta_bins = Counter()
    total_edges = 0

    print(f"Analyzing {len(graph_files)} graphs...")
    for f in graph_files:
        graph = torch.load(f, weights_only=False)
        if graph.edge_attr is None or graph.edge_attr.shape[0] == 0:
            continue

        # Edges are duplicated for undirected, so we only look at one direction
        unique_edges_mask = graph.edge_index[0] < graph.edge_index[1]
        edge_attrs = graph.edge_attr[unique_edges_mask]
        
        if edge_attrs.shape[0] == 0:
            continue

        total_edges += edge_attrs.shape[0]

        # is_sequential is column 0
        sequential_edges = edge_attrs[:, 0] == 1
        edge_type_counts['sequential'] += sequential_edges.sum().item()

        # is_shared_attribute is column 1
        shared_attr_edges = edge_attrs[:, 1] == 1
        edge_type_counts['shared_attribute'] += shared_attr_edges.sum().item()
        
        # Count edges that are both
        both_edges = sequential_edges & shared_attr_edges
        edge_type_counts['sequential_and_shared_attribute'] += both_edges.sum().item()

        # Analyze bins ONLY for sequential edges, as per preprocessing logic
        seq_edge_attrs = edge_attrs[sequential_edges]
        if seq_edge_attrs.shape[0] > 0:
            # Column 2: time_delta_bin, Column 3: geo_distance_bin, Column 4: amount_delta_bin
            # Bins are 1-indexed, 0 is for NO_VALUE
            time_delta_bins.update(seq_edge_attrs[:, 2].tolist())
            geo_dist_bins.update(seq_edge_attrs[:, 3].tolist())
            amt_delta_bins.update(seq_edge_attrs[:, 4].tolist())

    print("\n--- Edge Distribution Analysis ---")
    print(f"Total unique edges in training set: {total_edges}")
    
    print("\nEdge Type Counts:")
    for edge_type, count in edge_type_counts.items():
        percentage = (count / total_edges) * 100 if total_edges > 0 else 0
        print(f"  - {edge_type}: {count} ({percentage:.2f}%)")

    def print_bin_dist(name, counter):
        print(f"\nDistribution for {name} (on sequential edges):")
        total_in_dist = sum(counter.values())
        if total_in_dist == 0:
            print("  No sequential edges found with this feature.")
            return
            
        # Bin 0 is "NO_VALUE", should not appear for sequential edges but we check anyway
        if 0 in counter:
            print(f"  - Bin 0 (NO_VALUE): {counter.pop(0)} edges")

        for bin_val, count in sorted(counter.items()):
            percentage = (count / total_in_dist) * 100 if total_in_dist > 0 else 0
            print(f"  - Bin {bin_val}: {count} ({percentage:.2f}%)")

    print_bin_dist("Time Delta Bins", time_delta_bins)
    print_bin_dist("Geo Distance Bins", geo_dist_bins)
    print_bin_dist("Amount Delta Bins", amt_delta_bins)

if __name__ == '__main__':
    analyze_edge_distribution()
