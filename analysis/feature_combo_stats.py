import hydra
from omegaconf import DictConfig
import torch
from collections import Counter
import numpy as np
import pickle
import os

from src.datasets.transaction_dataset import TransactionDataModule, TransactionDatasetInfos
from src import utils

@hydra.main(version_base='1.1', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    # 1. Load the training dataloader
    datamodule = TransactionDataModule(cfg)
    train_loader = datamodule.train_dataloader()
    dataset_infos = TransactionDatasetInfos(datamodule, cfg)

    # 2. Initialize containers for statistics
    unique_node_combos = set()
    unique_edge_combos = set()
    
    node_combo_counts = Counter()
    edge_combo_counts = Counter()

    node_field_counts = [Counter() for _ in range(len(dataset_infos.num_node_features))]
    edge_field_counts = [Counter() for _ in range(len(dataset_infos.num_edge_features))]

    total_nodes = 0
    total_edges = 0

    print("Starting feature combination analysis...")
    # 3. Iterate over all batches in the training loader
    for i, batch in enumerate(train_loader):
        print(f"Processing batch {i+1}/{len(train_loader)}")
        dense_data, node_mask = utils.to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        # Process node features
        for i in range(dense_data.X.shape[0]): # Iterate over batch
            for j in range(dense_data.X.shape[1]): # Iterate over nodes
                if node_mask[i, j]:
                    total_nodes += 1
                    node_features = tuple(dense_data.X[i, j].tolist())
                    unique_node_combos.add(node_features)
                    node_combo_counts[node_features] += 1
                    for k in range(len(node_features)):
                        node_field_counts[k][node_features[k]] += 1

        # Process edge features
        for i in range(dense_data.E.shape[0]): # Iterate over batch
            for j in range(dense_data.E.shape[1]): # Iterate over source nodes
                for k in range(dense_data.E.shape[2]): # Iterate over target nodes
                    if node_mask[i, j] and node_mask[i, k] and j != k:
                        # Check if there is an edge
                        if dense_data.E[i, j, k].sum() != 0:
                            total_edges += 1
                            edge_features = tuple(dense_data.E[i, j, k].tolist())
                            unique_edge_combos.add(edge_features)
                            edge_combo_counts[edge_features] += 1
                            for l in range(len(edge_features)):
                                edge_field_counts[l][edge_features[l]] += 1
    
    print("\n--- Feature Combination Analysis Results ---")

    # 4. Print statistics
    print(f"\nTotal nodes processed: {total_nodes}")
    print(f"Total edges processed: {total_edges}")

    num_unique_nodes = len(unique_node_combos)
    num_unique_edges = len(unique_edge_combos)
    print(f"\nNumber of unique node feature combinations: {num_unique_nodes}")
    print(f"Number of unique edge feature combinations: {num_unique_edges}")

    total_possible_nodes = np.prod(dataset_infos.node_feature_dims)
    total_possible_edges = np.prod(dataset_infos.edge_feature_dims)
    print(f"\nEstimated total combinatorial space for nodes: {total_possible_nodes}")
    print(f"Estimated total combinatorial space for edges: {total_possible_edges}")

    print(f"\nNode combination sparsity: {num_unique_nodes / total_possible_nodes:.6f} ({num_unique_nodes}/{total_possible_nodes})")
    print(f"Edge combination sparsity: {num_unique_edges / total_possible_edges:.6f} ({num_unique_edges}/{total_possible_edges})")

    print("\nTop 20 most frequent node combinations:")
    for combo, count in node_combo_counts.most_common(20):
        print(f"  {combo}: {count}")

    print("\nTop 20 most frequent edge combinations:")
    for combo, count in edge_combo_counts.most_common(20):
        print(f"  {combo}: {count}")

    print("\nNode feature per-field distributions:")
    for i, (name, counts) in enumerate(zip(dataset_infos.num_node_features.keys(), node_field_counts)):
        print(f"  Field '{name}':")
        for value, count in counts.most_common(10):
            print(f"    Value {value}: {count} ({(count/total_nodes)*100:.2f}%)")

    print("\nEdge feature per-field distributions:")
    for i, (name, counts) in enumerate(zip(dataset_infos.num_edge_features.keys(), edge_field_counts)):
        print(f"  Field '{name}':")
        for value, count in counts.most_common(10):
            print(f"    Value {value}: {count} ({(count/total_edges)*100:.2f}%)")

    # 5. Save statistics
    stats = {
        'unique_node_combos': unique_node_combos,
        'unique_edge_combos': unique_edge_combos,
        'node_combo_counts': node_combo_counts,
        'edge_combo_counts': edge_combo_counts,
        'node_field_counts': node_field_counts,
        'edge_field_counts': edge_field_counts,
        'total_nodes': total_nodes,
        'total_edges': total_edges,
        'total_possible_nodes': total_possible_nodes,
        'total_possible_edges': total_possible_edges
    }
    
    save_path = "analysis/feature_combo_stats.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(stats, f)
    print(f"\nSaved statistics to {save_path}")


if __name__ == "__main__":
    main()
