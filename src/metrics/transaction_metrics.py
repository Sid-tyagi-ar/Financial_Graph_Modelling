from typing import List
from collections import Counter
import numpy as np
import torch
import wandb
from torch_geometric.data import Data

from src.analysis.dist_helper import compute_mmd, gaussian_tv
from src.analysis.visualization import NonMolecularVisualization


class TransactionSamplingMetrics:
    def __init__(self, datamodule, dataset_infos):
        self.train_dataloader = datamodule.train_dataloader()
        self.dataset_infos = dataset_infos
        self.visualizer = NonMolecularVisualization(dataset_infos=self.dataset_infos)

        print("Precomputing training set statistics for sampling metrics...")
        self.train_stats = self._compute_stats_from_loader(self.train_dataloader)
        print("Done.")

    def _compute_stats_from_loader(self, dataloader):
        degree_sequences = []
        edge_type_counts = []
        time_delta_counts = []

        for batch in dataloader:
            for graph_data in batch.to_data_list():
                if graph_data.x is None or graph_data.x.shape[0] == 0:
                    continue
                
                g = self.visualizer.to_networkx(graph_data)
                
                degrees = [d for _, d in g.degree()]
                degree_sequences.append(np.array(degrees))

                edge_counts_graph = Counter()
                time_counts_graph = Counter()
                for _, _, edge_data in g.edges(data=True):
                    if edge_data.get('is_root_txn_edge') == 1:
                        edge_counts_graph['root_to_txn'] += 1
                    if edge_data.get('is_txn_txn_edge') == 1:
                        edge_counts_graph['txn_to_txn_shared'] += 1
                    time_bin = edge_data.get('time_delta_bin', 0)
                    if time_bin > 0:
                        edge_counts_graph['sequential'] += 1
                        time_counts_graph[time_bin] += 1
                edge_type_counts.append(edge_counts_graph)
                time_delta_counts.append(time_counts_graph)
        
        return {'degree': degree_sequences, 'edge_type': edge_type_counts, 'time_delta': time_delta_counts}

    def _compute_stats_from_generated(self, generated_graphs):
        graphs = []
        for graph_tensors in generated_graphs:
            node_feat, edge_feat = graph_tensors
            if node_feat.shape[0] == 0: continue

            # Reconstruct a PyG Data object to pass to the visualizer
            num_nodes = node_feat.shape[0]
            adj = (edge_feat.argmax(dim=-1) > 0)
            edge_index = adj.nonzero().t().contiguous()
            
            sparse_edge_attr = edge_feat[edge_index[0], edge_index[1]]
            
            data_obj = Data(x=node_feat, edge_index=edge_index, edge_attr=sparse_edge_attr)
            data_obj.num_nodes = num_nodes

            g = self.visualizer.to_networkx(data_obj)
            graphs.append(g)

        degree_sequences = [[d for _, d in g.degree()] for g in graphs]
        
        edge_type_counts = []
        time_delta_counts = []
        for g in graphs:
            edge_counts_graph = Counter()
            time_counts_graph = Counter()
            for _, _, edge_data in g.edges(data=True):
                if edge_data.get('is_root_txn_edge') == 1:
                    edge_counts_graph['root_to_txn'] += 1
                if edge_data.get('is_txn_txn_edge') == 1:
                    edge_counts_graph['txn_to_txn_shared'] += 1
                time_bin = edge_data.get('time_delta_bin', 0)
                if time_bin > 0:
                    edge_counts_graph['sequential'] += 1
                    time_counts_graph[time_bin] += 1
            edge_type_counts.append(edge_counts_graph)
            time_delta_counts.append(time_counts_graph)

        return {'degree': degree_sequences, 'edge_type': edge_type_counts, 'time_delta': time_delta_counts}

    def __call__(self, generated_graphs, name, epoch, val_counter, test=False):
        print(f"Computing sampling metrics for epoch {epoch}...")
        gen_stats = self._compute_stats_from_generated(generated_graphs)

        # Helper to convert list of counters to list of normalized numpy arrays
        def get_dist_list(counters, keys):
            dist_list = []
            for c in counters:
                total = sum(c.values())
                if total == 0:
                    dist_list.append(np.zeros(len(keys)))
                    continue
                dist = np.array([c.get(k, 0) for k in keys]) / total
                dist_list.append(dist)
            return dist_list

        # MMD for Degree Distribution
        degree_mmd = compute_mmd(self.train_stats['degree'], gen_stats['degree'], kernel=gaussian_tv)
        
        # MMD for Edge Type Distribution
        edge_type_keys = ['root_to_txn', 'txn_to_txn_shared', 'sequential']
        train_edge_dist = get_dist_list(self.train_stats['edge_type'], edge_type_keys)
        gen_edge_dist = get_dist_list(gen_stats['edge_type'], edge_type_keys)
        edge_type_mmd = compute_mmd(train_edge_dist, gen_edge_dist, kernel=gaussian_tv) if gen_edge_dist else -1.0

        # MMD for Time Delta Distribution
        time_delta_keys = [1, 2, 3, 4]
        train_time_dist = get_dist_list(self.train_stats['time_delta'], time_delta_keys)
        gen_time_dist = get_dist_list(gen_stats['time_delta'], time_delta_keys)
        time_delta_mmd = compute_mmd(train_time_dist, gen_time_dist, kernel=gaussian_tv) if gen_time_dist else -1.0

        results = {
            'degree_mmd': degree_mmd,
            'edge_type_mmd': edge_type_mmd,
            'time_delta_mmd': time_delta_mmd
        }

        print(f"Sampling metrics for epoch {epoch}:")
        for key, val in results.items():
            print(f"  {key}: {val:.4f}")
        
        wandb.log({f'sampling_test/{k}' if test else f'sampling/{k}': v for k, v in results.items()}, commit=False)

    def reset(self):
        pass

