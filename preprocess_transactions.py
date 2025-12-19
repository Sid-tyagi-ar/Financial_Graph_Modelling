"""
Preprocess transaction data into PyG graphs.

This script:
1. Reads CSV with transaction data
2. Creates graphs (root node + transaction nodes)
3. Quantizes continuous features into bins
4. Saves graphs as .pt files
5. Saves encoders and bin_info for later decoding

Input CSV columns expected:
- person_id: account identifier
- transaction_id: unique transaction ID
- amount: transaction amount
- merchant: merchant name
- category: transaction category
- timestamp: transaction time (unix or datetime)
- city_pop: city population size
- merch_lat, merch_lon: merchant coordinates
- is_fraud: 0/1 label (optional)

Output:
- processed_graphs/*.pt (PyG Data objects)
- encoders.pkl (LabelEncoders for categorical columns)
- bin_info.pkl (bin edges for quantization)
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import KBinsDiscretizer
import torch
from torch_geometric.data import Data
from haversine import haversine, Unit


class TransactionGraphPreprocessor:
    NO_VALUE_FOR_FEATURE_IDX = 0
    """
    Convert raw transaction CSV into PyG graphs with quantized features.
    """
    
    def __init__(self,
                 output_root: str = ".",
                 n_bins: int = 10,
                 train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15)):
        """
        Args:
            output_root: Directory to save processed graphs, encoders, bin_info
            n_bins: Number of bins for quantization of continuous features
            train_val_test_split: Fraction for train/val/test splits
        """
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        self.processed_dir = self.output_root / "processed_graphs"
        self.processed_dir.mkdir(exist_ok=True)
        
        self.n_bins = n_bins
        self.train_val_test_split = train_val_test_split
        
        # Will be populated during preprocessing
        self.encoders: Dict[str, LabelEncoder] = {}
        self.bin_info: Dict[str, np.ndarray] = {}
        self.discretizers: Dict[str, KBinsDiscretizer] = {}
    
    def preprocess(self, csv_path: str, fraud_only: bool = True):
        """
        Main preprocessing pipeline.
        """
        print(f"Loading CSV from {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} transactions")

        
        df = df[df['is_fraud'] == 1]
        print(f"Filtered to {len(df)} fraudulent transactions.")

        # Step 1: Encode categorical columns
        print("\nStep 1: Encoding categorical columns...")
        df = self._encode_categorical_columns(df)
        
        # Step 2: Quantize continuous columns
        print("Step 2: Quantizing continuous columns...")
        df = self._quantize_continuous_columns(df)

        # Step 3: Process demographic data for conditional labels (y)
        print("Step 3: Processing demographic data for labels...")
        person_df = self._process_demographics(df)
        
        # Step 4: Build graphs
        print("Step 4: Building graphs...")
        graphs = self._build_graphs(df, person_df)
        print(f"Built {len(graphs)} graphs")
        
        # Step 5: Split into train/val/test
        print("Step 5: Splitting into train/val/test...")
        train_graphs, val_graphs, test_graphs = self._split_graphs(graphs)
        print(f"  Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")
        
        # Step 6: Save graphs
        print("Step 6: Saving graphs...")
        self._save_graphs(train_graphs, val_graphs, test_graphs)
        
        # Step 7: Save encoders and bin info
        print("Step 7: Saving metadata...")
        self._save_metadata()
        
        print("\nPreprocessing complete!")
        print(f"  Graphs saved to: {self.processed_dir}")
        print(f"  Encoders saved to: {self.output_root / 'encoders.pkl'}")
        print(f"  Bin info saved to: {self.output_root / 'bin_info.pkl'}")
    
    def _encode_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical columns (merchant, category) using LabelEncoder.
        """
        categorical_cols = ['merchant', 'category']
        
        for col in categorical_cols:
            if col not in df.columns:
                print(f"  Warning: Column '{col}' not found in CSV")
                continue
            
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            self.encoders[col] = le
            
            n_unique = len(le.classes_)
            print(f"  {col}: {n_unique} unique values")
        
        return df
    
    def _quantize_continuous_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Quantize continuous columns into bins.
        """
        continuous_cols = {
            'amt': 'amt',
            'city_pop': 'city_pop',
            'merch_lat': 'merch_lat',
            'merch_long': 'merch_long',
            'unix_time': 'unix_time'
        }
        
        for csv_col, bin_name in continuous_cols.items():
            if csv_col not in df.columns:
                print(f"  Warning: Column '{csv_col}' not found in CSV")
                continue
            
            # Handle missing values
            col_data = df[csv_col].fillna(df[csv_col].median())
            
            # Quantize into bins
            discretizer = KBinsDiscretizer(
                n_bins=self.n_bins,
                encode='ordinal',
                strategy='quantile'
            )
            bins = discretizer.fit_transform(col_data.values.reshape(-1, 1)).astype(int).flatten()
            df[f'{csv_col}_bin'] = bins
            
            # Store bin edges for later dequantization
            self.bin_info[bin_name] = discretizer.bin_edges_[0]
            self.discretizers[bin_name] = discretizer
            
            print(f"  {csv_col}: {self.n_bins} bins")
        
        return df
    
    def _process_demographics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process demographic data to create labels for each person.
        """
        person_df = df.groupby('cc_num').first().reset_index()

        # Encode categorical demographic features
        demographic_cols = ['gender', 'city', 'state', 'job']
        for col in demographic_cols:
            if col in person_df.columns:
                le = LabelEncoder()
                person_df[f'{col}_encoded'] = le.fit_transform(person_df[col].astype(str))
                self.encoders[f'person_{col}'] = le
                print(f"  Demographic '{col}': {len(le.classes_)} unique values")

        # Calculate and bin age
        if 'dob' in person_df.columns:
            person_df['dob'] = pd.to_datetime(person_df['dob'])
            current_year = datetime.now().year
            person_df['age'] = current_year - person_df['dob'].dt.year
            
            age_discretizer = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='quantile')
            person_df['age_bin'] = age_discretizer.fit_transform(person_df['age'].values.reshape(-1, 1)).astype(int).flatten()
            self.bin_info['person_age'] = age_discretizer.bin_edges_[0]
            self.discretizers['person_age'] = age_discretizer
            print(f"  Demographic 'age': {self.n_bins} bins")

        return person_df

    def _build_graphs(self, df: pd.DataFrame, person_df: pd.DataFrame) -> List[Data]:
        """
        Build PyG graphs: one graph per person with root node + transaction nodes.
        """
        graphs = []
        person_groups = df.groupby('cc_num')
        
        person_df.set_index('cc_num', inplace=True)

        for person_id, group in person_groups:
            person_info = person_df.loc[person_id]
            
            # Create the label vector y
            y_features = []
            if 'gender_encoded' in person_info:
                y_features.append(person_info['gender_encoded'])
            if 'city_encoded' in person_info:
                y_features.append(person_info['city_encoded'])
            if 'state_encoded' in person_info:
                y_features.append(person_info['state_encoded'])
            if 'job_encoded' in person_info:
                y_features.append(person_info['job_encoded'])
            if 'age_bin' in person_info:
                y_features.append(person_info['age_bin'])
            
            y = torch.tensor([y_features], dtype=torch.long)
            
            graph = self._build_single_graph(group, y)
            if graph is not None:
                graphs.append(graph)
        
        return graphs

    def _build_single_graph(self, transactions_df: pd.DataFrame, y_label: torch.Tensor) -> Optional[Data]:
        """
        Build a single "rootless" graph for one person's transactions.
        Each node is a transaction, and edges represent relationships between them.
        """
        # Sort by time to create sequential edges
        transactions_df = transactions_df.sort_values('unix_time').reset_index(drop=True)

        n_transactions = len(transactions_df)
        if n_transactions < 2:  # Need at least 2 transactions to form an edge
            return None
        
        # ===== Node Features =====
        # Layout: [amt_bin, merch_id, cat_id, city_pop_bin, merch_lat_bin, merch_lon_bin, unix_time_bin]
        node_features = []
        for _, row in transactions_df.iterrows():
            feat = [
                int(row.get('amt_bin', 0)),
                int(row.get('merchant_encoded', 0)),
                int(row.get('category_encoded', 0)),
                int(row.get('city_pop_bin', 0)),
                int(row.get('merch_lat_bin', 0)),
                int(row.get('merch_lon_bin', 0)),
                int(row.get('unix_time_bin', 0))
            ]
            node_features.append(feat)
        
        X = torch.tensor(node_features, dtype=torch.long)

        # ===== Pre-calculate deltas for edge features =====
        time_deltas = transactions_df['unix_time'].diff().dropna().abs()
        amount_deltas = transactions_df['amt'].diff().dropna().abs()
        
        # Haversine distance
        geo_distances = []
        for i in range(len(transactions_df) - 1):
            lat1, lon1 = transactions_df.loc[i, ['merch_lat', 'merch_long']]
            lat2, lon2 = transactions_df.loc[i+1, ['merch_lat', 'merch_long']]
            geo_distances.append(haversine((lat1, lon1), (lat2, lon2), unit=Unit.KILOMETERS))
        
        # Bin the deltas
        time_delta_bins = pd.cut(time_deltas, bins=self.n_bins, labels=False, include_lowest=True)
        amount_delta_bins = pd.cut(amount_deltas, bins=self.n_bins, labels=False, include_lowest=True)
        geo_distance_bins = pd.cut(pd.Series(geo_distances), bins=self.n_bins, labels=False, include_lowest=True)

        # ===== Edge Features =====
        # Layout: [is_sequential, is_shared_attribute, time_delta_bin, geo_distance_bin, amount_delta_bin]
        edge_list = []
        edge_attrs_list = []

        merchants = transactions_df['merchant_encoded'].values
        categories = transactions_df['category_encoded'].values

        for i in range(n_transactions):
            for j in range(i + 1, n_transactions):
                is_sequential = (j == i + 1)
                is_shared_attribute = (merchants[i] == merchants[j] or categories[i] == categories[j])

                # Only add an edge if there's a relationship
                if not is_sequential and not is_shared_attribute:
                    continue

                time_delta_bin = time_delta_bins.iloc[i-1] if is_sequential and i > 0 else self.NO_VALUE_FOR_FEATURE_IDX
                geo_dist_bin = geo_distance_bins.iloc[i-1] if is_sequential and i > 0 else self.NO_VALUE_FOR_FEATURE_IDX
                amt_delta_bin = amount_delta_bins.iloc[i-1] if is_sequential and i > 0 else self.NO_VALUE_FOR_FEATURE_IDX

                edge_attr = [
                    int(is_sequential),
                    int(is_shared_attribute),
                    int(time_delta_bin) + 1,  # +1 to reserve 0 for NO_VALUE
                    int(geo_dist_bin) + 1,
                    int(amt_delta_bin) + 1
                ]
                
                # Add edges in both directions for undirected graph
                edge_list.append([i, j])
                edge_attrs_list.append(edge_attr)
                edge_list.append([j, i])
                edge_attrs_list.append(edge_attr)

        if not edge_list:
            return None # No edges were created

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs_list, dtype=torch.long)
        
        # Create PyG Data object
        graph = Data(
            x=X,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y_label,
            num_nodes=n_transactions
        )
        
        return graph
    
    def _split_graphs(self, graphs: List[Data]) -> Tuple[List[Data], List[Data], List[Data]]:
        """
        Split graphs into train/val/test sets.
        """
        n_total = len(graphs)
        train_frac, val_frac, test_frac = self.train_val_test_split
        
        n_train = int(n_total * train_frac)
        n_val = int(n_total * val_frac)
        
        # Shuffle
        indices = np.random.permutation(n_total)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        train_graphs = [graphs[i] for i in train_idx]
        val_graphs = [graphs[i] for i in val_idx]
        test_graphs = [graphs[i] for i in test_idx]
        
        return train_graphs, val_graphs, test_graphs
    
    def _save_graphs(self, train_graphs: List[Data], val_graphs: List[Data], test_graphs: List[Data]):
        """
        Save graphs to disk as .pt files in train/val/test subdirectories.
        """
        # Create subdirectories
        train_dir = self.processed_dir / "train"
        val_dir = self.processed_dir / "val"
        test_dir = self.processed_dir / "test"
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)
        test_dir.mkdir(exist_ok=True)

        # Save graphs
        for i, graph in enumerate(train_graphs):
            torch.save(graph, train_dir / f"graph_{i:06d}.pt")
        
        for i, graph in enumerate(val_graphs):
            torch.save(graph, val_dir / f"graph_{i:06d}.pt")
            
        for i, graph in enumerate(test_graphs):
            torch.save(graph, test_dir / f"graph_{i:06d}.pt")
            
        total_graphs = len(train_graphs) + len(val_graphs) + len(test_graphs)
        print(f"  Saved {total_graphs} graphs to {self.processed_dir} in train/val/test folders")
    
    def _save_metadata(self):
        """
        Save encoders and bin_info for later use.
        """
        # Add bin info for new delta features
        self.bin_info['time_delta_bins'] = self.n_bins + 1
        self.bin_info['geo_distance_bins'] = self.n_bins + 1
        self.bin_info['amount_delta_bins'] = self.n_bins + 1

        encoders_path = self.output_root / "encoders.pkl"
        bininfo_path = self.output_root / "bin_info.pkl"
        
        with open(encoders_path, 'wb') as f:
            pickle.dump(self.encoders, f)
        
        with open(bininfo_path, 'wb') as f:
            pickle.dump(self.bin_info, f)
        
        print(f"  Saved encoders to {encoders_path}")
        print(f"  Saved bin_info to {bininfo_path}")


def create_sample_csv(output_path: str = "sample_transactions.csv", n_persons: int = 100, n_txns_per_person: int = 20):
    """
    Create a sample CSV for testing the preprocessor.
    """
    np.random.seed(42)
    
    merchants = ['Amazon', 'Walmart', 'Target', 'Starbucks', 'Shell', 'BestBuy', 'Costco']
    categories = ['Electronics', 'Groceries', 'Gas', 'Food', 'Entertainment']
    
    data = []
    for person_id in range(n_persons):
        for txn_id in range(n_txns_per_person):
            data.append({
                'person_id': person_id,
                'transaction_id': f"{person_id}_{txn_id}",
                'amount': np.random.uniform(10, 1000),
                'merchant': np.random.choice(merchants),
                'category': np.random.choice(categories),
                'timestamp': 1700000000 + (txn_id * 86400),  # unix timestamps
                'city_pop': np.random.randint(10000, 5000000),
                'merch_lat': np.random.uniform(-90, 90),
                'merch_lon': np.random.uniform(-180, 180),
                'is_fraud': 1 if np.random.random() < 0.05 else 0
            })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Created sample CSV: {output_path}")
    print(f"  {len(df)} transactions across {n_persons} persons")
    return output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess transaction data into PyG graphs')
    parser.add_argument('--csv_path', type=str, default='transactions.csv',
                        help='Path to input CSV file')
    parser.add_argument('--output_root', type=str, default='.',
                        help='Root directory for output (graphs, encoders, bin_info)')
    parser.add_argument('--n_bins', type=int, default=10,
                        help='Number of bins for quantization')
    parser.add_argument('--create_sample', action='store_true',
                        help='Create sample CSV for testing')
    parser.add_argument('--fraud_only', action='store_true',
                        help='Only process fraudulent transactions.')

    args = parser.parse_args()

    # Create sample if needed
    if args.create_sample:
        args.csv_path = create_sample_csv()

    # Preprocess
    preprocessor = TransactionGraphPreprocessor(
        output_root=args.output_root,
        n_bins=args.n_bins
    )
    preprocessor.preprocess(args.csv_path, fraud_only=args.fraud_only)

    print("\nUsage with DiGress:")
    print(f"python main.py dataset=transaction output_root={args.output_root}")
