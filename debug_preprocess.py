"""
Run this script AFTER preprocessing to verify your graphs are correct.
Usage: python debug_after_preprocess.py
"""

import os
import sys
import pickle
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import numpy as np
from glob import glob
from collections import Counter

def analyze_preprocessed_graphs(datadir='.'):
    """Analyze the preprocessed graphs to check diversity and correctness."""
    
    print("="*80)
    print("ANALYZING PREPROCESSED GRAPHS")
    print("="*80)
    
    # Load metadata
    encoders_path = os.path.join(datadir, 'encoders.pkl')
    bin_info_path = os.path.join(datadir, 'bin_info.pkl')
    
    if not os.path.exists(encoders_path):
        print(f"❌ Encoders not found at {encoders_path}")
        return
    
    with open(encoders_path, 'rb') as f:
        encoders = pickle.load(f)
    with open(bin_info_path, 'rb') as f:
        bin_info = pickle.load(f)
    
    # Load a sample of graphs from train split
    train_dir = os.path.join(datadir, 'processed_graphs', 'train')
    graph_files = sorted(glob(os.path.join(train_dir, '*.pt')))[:50]  # First 50 graphs
    
    if not graph_files:
        print(f"❌ No graphs found in {train_dir}")
        return
    
    print(f"\n1. DATASET OVERVIEW")
    print(f"   Found {len(graph_files)} train graphs (analyzing first 50)")
    
    graphs = [torch.load(f, weights_only=False) for f in graph_files[:10]]
    
    # Analyze graph properties
    print(f"\n2. GRAPH STRUCTURE")
    sizes = [g.num_nodes for g in graphs]
    edge_counts = [g.edge_index.size(1) for g in graphs]
    
    print(f"   Number of nodes:")
    print(f"     Min: {min(sizes)}, Max: {max(sizes)}, Mean: {np.mean(sizes):.1f}")
    
    print(f"   Number of edges:")
    print(f"     Min: {min(edge_counts)}, Max: {max(edge_counts)}, Mean: {np.mean(edge_counts):.1f}")
    
    # Analyze feature diversity
    print(f"\n3. NODE FEATURE DIVERSITY")
    
    # Collect all unique node feature combinations
    all_node_features = []
    for g in graphs:
        for i in range(g.num_nodes):
            node_tuple = tuple(g.x[i].tolist())
            all_node_features.append(node_tuple)
    
    unique_nodes = len(set(all_node_features))
    print(f"   Total nodes analyzed: {len(all_node_features)}")
    print(f"   Unique node types: {unique_nodes}")
    print(f"   Diversity ratio: {unique_nodes / len(all_node_features):.2%}")
    
    if unique_nodes / len(all_node_features) < 0.1:
        print(f"   ⚠️ WARNING: Low node diversity - many nodes are identical")
        print(f"              This is normal for fraud-only data but may cause low losses")
    
    # Analyze specific features
    print(f"\n4. FEATURE DISTRIBUTION (RAW INDICES)")
    
    # Amount bins
    amounts = [g.x[i, 0].item() for g in graphs for i in range(1, g.num_nodes)]  # Skip root
    amount_dist = Counter(amounts)
    print(f"   Amount bins: {len(amount_dist)} unique values")
    print(f"   Top 3 amounts: {amount_dist.most_common(3)}")
    
    # Merchants
    merchants = [g.x[i, 1].item() for g in graphs for i in range(1, g.num_nodes)]
    merchant_dist = Counter(merchants)
    print(f"   Merchants: {len(merchant_dist)} unique values")
    print(f"   Top 3 merchants: {merchant_dist.most_common(3)}")
    
    # Categories
    categories = [g.x[i, 2].item() for g in graphs for i in range(1, g.num_nodes)]
    category_dist = Counter(categories)
    print(f"   Categories: {len(category_dist)} unique values")
    print(f"   Top 3 categories: {category_dist.most_common(3)}")
    
    # Check edge features
    print(f"\n5. EDGE FEATURE DISTRIBUTION")
    
    all_edge_features = []
    for g in graphs:
        for i in range(g.edge_attr.size(0)):
            edge_tuple = tuple(g.edge_attr[i].tolist())
            all_edge_features.append(edge_tuple)
    
    unique_edges = len(set(all_edge_features))
    print(f"   Total edges analyzed: {len(all_edge_features)}")
    print(f"   Unique edge types: {unique_edges}")
    print(f"   Diversity ratio: {unique_edges / len(all_edge_features):.2%}")
    
    # Analyze graph labels (y)
    print(f"\n6. GRAPH LABELS (y)")
    all_y = torch.stack([g.y for g in graphs])
    print(f"   Shape: {all_y.shape}")
    print(f"   Unique y combinations: {len(set([tuple(y.tolist()[0]) for y in all_y]))}")
    
    # Check for identical graphs
    print(f"\n7. GRAPH SIMILARITY CHECK")
    if len(graphs) > 1:
        # Compare first two graphs
        g1, g2 = graphs[0], graphs[1]
        
        if g1.num_nodes == g2.num_nodes:
            node_similarity = (g1.x == g2.x).float().mean().item()
            print(f"   Node similarity (graph 0 vs 1): {node_similarity:.2%}")
            
            if node_similarity > 0.9:
                print(f"   ⚠️ WARNING: Graphs are very similar!")
                print(f"              This suggests limited diversity in fraud patterns")
        else:
            print(f"   Graph sizes differ: {g1.num_nodes} vs {g2.num_nodes} nodes")
    
    # Feature dimensions for model
    print(f"\n8. COMPUTED DIMENSIONS FOR MODEL")
    
    num_node_features = {
        'type': 2,
        'amount': bin_info['amt'].shape[0] - 1,
        'merchant': len(encoders['merchant'].classes_),
        'category': len(encoders['category'].classes_)
    }
    
    num_edge_features = {
        'is_root_txn': 2,
        'is_txn_txn': 2,
        'city_pop': bin_info['city_pop_total_bins'],
        'merch_lat': bin_info['merch_lat_total_bins'],
        'merch_lon': bin_info['merch_lon_total_bins'],
        'unix_time': bin_info['unix_time_total_bins'],
        'shared_merch_cat': 2
    }
    
    print(f"   Node features:")
    for k, v in num_node_features.items():
        print(f"     {k}: {v}")
    total_node = sum(num_node_features.values())
    print(f"   Total (after one-hot): {total_node}")
    
    print(f"\n   Edge features:")
    for k, v in num_edge_features.items():
        print(f"     {k}: {v}")
    total_edge = sum(num_edge_features.values())
    print(f"   Total (after one-hot): {total_edge}")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("9. RECOMMENDATIONS")
    print(f"{'='*80}")
    
    recommendations = []
    
    if unique_nodes / len(all_node_features) < 0.05:
        recommendations.append("⚠️ Very low node diversity detected:")
        recommendations.append("   - This will cause low training losses (expected for fraud-only)")
        recommendations.append("   - Consider: Are there enough diverse fraud patterns?")
        recommendations.append("   - Model will generate very similar graphs to training data")
    
    if len(graphs) < 50:
        recommendations.append(f"⚠️ Small dataset: only {len(graphs)} graphs")
        recommendations.append("   - Minimum recommended: 100+ graphs for stable training")
        recommendations.append("   - Consider collecting more fraud transactions")
    
    if len(merchant_dist) < 3:
        recommendations.append("⚠️ Very few unique merchants - model may not generalize")
    
    if len(category_dist) < 3:
        recommendations.append("⚠️ Very few unique categories - model may not generalize")
    
    if not recommendations:
        recommendations.append("✅ Dataset looks reasonable for fraud-only generation")
        recommendations.append("   Low losses are expected due to homogeneous fraud patterns")
    
    for rec in recommendations:
        print(f"   {rec}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")
    
    return {
        'num_graphs': len(graph_files),
        'unique_nodes': unique_nodes,
        'node_diversity': unique_nodes / len(all_node_features),
        'unique_edges': unique_edges,
        'total_node_dim': total_node,
        'total_edge_dim': total_edge
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='.', 
                        help='Directory containing processed_graphs/')
    args = parser.parse_args()
    
    results = analyze_preprocessed_graphs(args.datadir)
    print("\nRun this after preprocessing completes:")
    print(f"python debug_after_preprocess.py --datadir {args.datadir}")
