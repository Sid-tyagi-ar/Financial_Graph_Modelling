"""
Transaction dataset and dataset-info utilities for DiGress transactions data.
Module overview
----------------
This module provides PyTorch Geometric Dataset and helper classes to load,
process and describe a transactions graph dataset for the DiGress project.
It supports two loading modes:
- "new_preproc" mode: expects a preprocessed CSV and an encoders.pkl produced
    by the newer preprocessing pipeline (stored under data/preprocessed/ and
    data/encoders/). In this mode the Dataset will lazily load processed
    per-person graph files from the `processed` directory (train/val/test).
- "legacy" mode: falls back to the old format that uses pickled graphs under
    processed_graphs/<split>/ and two files encoders.pkl and bin_info.pkl in
    the dataset folder.
Files expected (new_preproc)
- data/encoders/encoders.pkl  : dict-like encoder objects and metadata used to
                                                                map categorical values and bin sizes.
- data/preprocessed/processed.csv : flat table of transactions with one row
                                                                        per transaction. Required columns used:
                                                                        cc_num, unix_time, merchant_id, amt_bin,
                                                                        merchant_cluster, category_small_id, city_id,
                                                                        time_bucket, merch_lat, merch_long,
                                                                        gender_id, job_id, age_bucket, gender_id, city_id, job_id, age_bucket
- processed/<split>/graph_{i}.pt : per-person PyG Data objects created by
                                                                        `process()` (one file per person/graph).
Files expected (legacy)
- processed_graphs/<split>/*.pt : raw PyG Data objects.
- encoders.pkl                 : legacy sklearn LabelEncoder-like encoders.
- bin_info.pkl                 : numeric bin boundaries and counts used for
                                                                 discretization (amt, time, lat/lon, etc).
Classes
-------
TransactionDataset(split, root, datadir, transform=None, pre_transform=None, pre_filter=None)
        PyG Dataset implementing loading and optional on-disk processing for
        per-person transaction graphs.
        Behavior:
        - Detects the presence of the new preprocessing artifacts (encoders.pkl in
            data/encoders). If found the dataset enters new_preproc mode and expects
            a processed CSV file to generate individual graphs on disk via `process()`.
        - In new_preproc mode, attributes describing categorical dims are computed
            from the saved encoders and stored as:
                dims_node_features, dims_edge_features, dims_y_features, y_feature_dims
            and `processed_files` is resolved from processed/<split>/*.pt.
        - In legacy mode, legacy encoders and bin_info are loaded and equivalent
            dimension dictionaries are computed; raw graph file list is resolved from
            processed_graphs/<split>.
        Key attributes:
        - split : one of 'train', 'val', 'test'
        - datadir, root : dataset base paths used to locate encoders and CSVs
        - new_preproc : bool, whether the new preprocessing artifacts were found
        - dims_node_features : dict mapping node-field names -> number of categories
        - dims_edge_features : dict mapping edge-field names -> number of categories
        - dims_y_features    : dict mapping target-field names -> number of classes
        - processed_files / raw_files : lists of .pt files used by `get()`.
        Important methods:
        - raw_file_names / processed_file_names : properties used by PyG framework.
        - process() :
                If new_preproc==True, reads the processed CSV and:
                    * groups transactions by `cc_num` (card / person id),
                    * sorts by unix_time,
                    * builds node features (merchant_id, amt_bin, merchant_cluster, category_small_id, city_id, time_bucket),
                    * builds edge list and edge attributes by linking transactions that are
                        sequential, share merchant cluster, or are geographically close,
                        and stores symmetrical edges (both directions),
                    * computes time_delta bins (via pandas.cut) and geographic distance
                        (using haversine) to populate edge attributes,
                    * stores per-person graphs into processed/<split>/graph_{i}.pt with a
                        hash-based split assignment (70% train, 20% val, 10% test).
                Note: graphs with fewer than 2 transactions or with no qualifying edges
                are skipped.
        - len() : returns number of available graph files for the active mode.
        - get(idx) : loads and returns a PyG Data object:
                - In new_preproc mode loads the processed file from processed_files.
                - In legacy mode converts raw graph fields to the expected types and
                    returns a Data(x=final_x, edge_index=..., edge_attr=..., y=...).
        Data object format (created/returned)
        - x : LongTensor of shape (num_nodes, num_node_fields) containing categorical
                    indices for node-level fields.
        - edge_index : LongTensor (2, num_edges) as per PyG conventions.
        - edge_attr : LongTensor (num_edges, num_edge_fields) containing categorical
                                    indices for edge-level fields.
        - y : LongTensor (1, num_y_fields) with person-level categorical targets.
        - num_nodes : integer number of nodes in the graph.
TransactionDataModule(cfg)
        Lightweight datamodule that wraps TransactionDataset for train/val/test
        usage and integrates with the project AbstractDataModule.
        Behavior:
        - Resolves datadir with hydra.get_original_cwd when available so running
            inside Hydra works correctly.
        - Instantiates TransactionDataset for each split and registers them with
            the parent AbstractDataModule via `prepare_data`.
TransactionDatasetInfos(datamodule, cfg)
        Metadata and helper for model construction and input/output dimensionality.
        Responsibilities:
        - Detects preprocessing mode (new vs legacy) similarly to TransactionDataset
            and loads encoders/bin_info accordingly.
        - Exposes:
                - num_node_features : mapping name -> categories (same as dims_node_features)
                - num_edge_features : mapping name -> categories
                - num_y_features    : mapping name -> classes
                - node_feature_dims, edge_feature_dims, y_feature_dims : lists of dims
                - node_types, edge_types : delegated to datamodule node/edge helpers
                - n_nodes : array/list of node count distribution from datamodule.node_counts()
                - max_n_nodes, num_classes : summary metrics
                - nodes_dist : DistributionNodes built over n_nodes (used for sampling)
        - compute_input_output_dims(datamodule, extra_features, domain_features):
                Estimates model input and output dimensionalities using:
                - a single example batch from datamodule.train_dataloader()
                - the configured embedding sizes:
                        cfg.model.hidden_dims.x_emb_dim
                        cfg.model.hidden_dims.e_emb_dim
                        cfg.model.hidden_dims.y_emb_dim
                - counts of categorical fields (differs between new and legacy modes)
                - additional extra feature modules (extra_features, domain_features),
                    which are called with a small dummy noisy_data structure to compute
                    appended continuous feature sizes for X, E and y.
                Sets:
                - self.input_dims : dict with 'X','E','y' containing projected input dims
                - self.output_dims: dict with 'X','E','y' containing output space dims
                                                        (sums of categorical dims per type).
Notes and implementation details
-------------------------------
- Geographic distance threshold and time delta binning logic live in
    TransactionDataset.process(); the threshold used is 40 km for "is_close_geo".
- Edge attributes include a reserved bin offset for time_delta_bin (`+1`)
    to avoid using 0 for "no information" vs encoded bins; check the preprocess
    logic if adapting to another encoder scheme.
- The module uses haversine for distance computation and pandas.cut for time
    delta binning; these dependencies must be present in the environment.
- For reproducible splits the dataset uses hash(cc_num) % 10 to assign train/val/test.
- All categorical fields are represented as integer indices (dtype long) and
    one-hot or learned embeddings are expected downstream by the model.
Example (high level)
--------------------
- Place encoders.pkl and processed.csv as described, call TransactionDataset(...).
- If you want to create processed/<split>/graph_*.pt from processed.csv, call
    dataset.process() once; the process method creates per-split processed files.
- Use TransactionDatasetInfos to compute model input/output sizes before model init.


"""
import os
import pickle
from typing import List

import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data
from glob import glob
from tqdm import tqdm
from haversine import haversine


from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from src.diffusion.distributions import DistributionNodes
from src import utils


class TransactionDataset(Dataset):
    def __init__(self, split: str, root: str, datadir: str, transform=None, pre_transform=None, pre_filter=None):
        self.split = split
        self.datadir = datadir
        self.new_preproc = False
        
        try:
            # This fails when running outside of hydra
            from hydra.utils import get_original_cwd
            new_encoder_path = os.path.join(get_original_cwd(), 'data', 'encoders', 'encoders.pkl')
        except (ValueError, ImportError):
            new_encoder_path = os.path.join('data', 'encoders', 'encoders.pkl')

        if os.path.exists(new_encoder_path):
            self.new_preproc = True
            import pandas as pd
            try:
                from hydra.utils import get_original_cwd
                csv_path = os.path.join(get_original_cwd(), 'data', 'preprocessed', 'processed.csv')
            except (ValueError, ImportError):
                csv_path = os.path.join('data', 'preprocessed', 'processed.csv')
            df = pd.read_csv(csv_path, usecols=['cc_num'])
            self.num_graphs = df['cc_num'].nunique()
        
        super().__init__(root, transform, pre_transform, pre_filter)

        if self.new_preproc:
            print("Found new preprocessed data. Using new data loading logic.")
            with open(new_encoder_path, 'rb') as f:
                self.encoders = pickle.load(f)
            
            self.dims_node_features = {
                'merchant_id': len(self.encoders['merchant_id_map']),
                'amt_bin': self.encoders['n_amt_bins'],
                'merchant_cluster': self.encoders['merchant_meta']['kmeans'].n_clusters + 1,
                'category_small_id': len(self.encoders['top_cats']) + 1,
                'city_id': len(self.encoders['city']),
                'time_bucket': 24 // self.encoders['time_bucket_hours']
            }
            self.dims_edge_features = {
                'is_sequential': 2,
                'same_merchant_cluster': 2,
                'time_delta_bin': 10,
                'is_close_geo': 2,
            }
            self.dims_y_features = {
                'gender': len(self.encoders['gender']),
                'city': len(self.encoders['city']),
                'job': len(self.encoders['job']),
                'age_bucket': len(self.encoders['age_bucket'])
            }
            self.y_feature_dims = list(self.dims_y_features.values())
            self.processed_files = sorted(glob(os.path.join(self.processed_dir, self.split, '*.pt')))

        else:
            print("WARNING: New encoders not found. Falling back to old data loading logic.")
            self.raw_files = sorted(glob(os.path.join(root, 'processed_graphs', self.split, '*.pt')))
            
            with open(os.path.join(self.datadir, 'encoders.pkl'), 'rb') as f:
                self.encoders = pickle.load(f)
            with open(os.path.join(self.datadir, 'bin_info.pkl'), 'rb') as f:
                self.bin_info = pickle.load(f)

            self.dims_node_features = {
                'amt_bin': self.bin_info['amt'].shape[0] - 1,
                'merch_id': len(self.encoders['merchant'].classes_),
                'cat_id': len(self.encoders['category'].classes_),
                'city_pop_bin': self.bin_info['city_pop'].shape[0] - 1,
                'merch_lat_bin': self.bin_info['merch_lat'].shape[0] - 1,
                'merch_lon_bin': self.bin_info['merch_long'].shape[0] - 1,
                'unix_time_bin': self.bin_info['unix_time'].shape[0] - 1
            }
            self.dims_edge_features = {
                'is_sequential': 2,
                'is_shared_attribute': 2,
                'time_delta_bin': self.bin_info['time_delta_bins'],
                'geo_distance_bin': self.bin_info['geo_distance_bins'],
                'amount_delta_bin': self.bin_info['amount_delta_bins']
            }
            self.dims_y_features = {
                'gender': len(self.encoders['person_gender'].classes_),
                'city': len(self.encoders['person_city'].classes_),
                'state': len(self.encoders['person_state'].classes_),
                'job': len(self.encoders['person_job'].classes_),
                'age': self.bin_info['person_age'].shape[0] - 1
            }
            self.y_feature_dims = list(self.dims_y_features.values())
    
    @property
    def raw_file_names(self) -> List[str]:
        if self.new_preproc:
            return ['processed.csv']
        else:
            return self.raw_files

    @property
    def processed_file_names(self) -> List[str]:
        if self.new_preproc:
            return [os.path.join(self.split, f'graph_{i}.pt') for i in range(self.num_graphs)]
        else:
            return []

    def download(self):
        pass

    def process(self):
        if not self.new_preproc:
            return

        import pandas as pd
        
        try:
            from hydra.utils import get_original_cwd
            csv_path = os.path.join(get_original_cwd(), 'data', 'preprocessed', 'processed.csv')
        except (ValueError, ImportError):
            csv_path = os.path.join('data', 'preprocessed', 'processed.csv')

        df = pd.read_csv(csv_path)
        
        person_groups = df.groupby('cc_num')
        
        train_dir = os.path.join(self.processed_dir, "train")
        val_dir = os.path.join(self.processed_dir, "val")
        test_dir = os.path.join(self.processed_dir, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        i = 0
        for person_id, group in tqdm(person_groups):
            group = group.sort_values('unix_time').reset_index(drop=True)

            n_transactions = len(group)
            if n_transactions < 2:
                continue

            # Node features
            node_features = group[['merchant_id', 'amt_bin', 'merchant_cluster', 'category_small_id', 'city_id', 'time_bucket']].values
            X = torch.tensor(node_features, dtype=torch.long)

            # Edge features
            edge_list = []
            edge_attrs_list = []
            
            time_deltas = group['unix_time'].diff().dropna().abs()
            time_delta_bins = pd.cut(time_deltas, bins=8, labels=False, include_lowest=True).fillna(0).astype(int)

            coords = group[['merch_lat', 'merch_long']].values

            for j in range(n_transactions):
                for k in range(j + 1, n_transactions):
                    is_sequential = (k == j + 1)
                    same_merchant_cluster = group.loc[j, 'merchant_cluster'] == group.loc[k, 'merchant_cluster']
                    
                    # Calculate geographical distance
                    coord1 = (coords[j, 0], coords[j, 1])
                    coord2 = (coords[k, 0], coords[k, 1])
                    geo_dist = haversine(coord1, coord2, unit='km')
                    is_close_geo = geo_dist < 40

                    if not is_sequential and not same_merchant_cluster and not is_close_geo:
                        continue

                    time_delta_bin = time_delta_bins.iloc[j-1] if is_sequential and j > 0 else 0

                    edge_attr = [
                        int(is_sequential),
                        int(same_merchant_cluster),
                        int(time_delta_bin) + 1,
                        int(is_close_geo),
                    ]
                    
                    edge_list.append([j, k])
                    edge_attrs_list.append(edge_attr)
                    edge_list.append([k, j])
                    edge_attrs_list.append(edge_attr)

            if not edge_list:
                continue

            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs_list, dtype=torch.long)

            # y features
            y_features = group[['gender_id', 'city_id', 'job_id', 'age_bucket']].iloc[0].values
            y = torch.tensor(y_features, dtype=torch.long).unsqueeze(0)

            graph = Data(
                x=X,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                num_nodes=n_transactions
            )
            
            split_val = hash(person_id) % 10
            if split_val < 7:
                torch.save(graph, os.path.join(train_dir, f"graph_{i}.pt"))
            elif split_val < 9:
                torch.save(graph, os.path.join(val_dir, f"graph_{i}.pt"))
            else:
                torch.save(graph, os.path.join(test_dir, f"graph_{i}.pt"))
            i += 1

    def len(self):
        if self.new_preproc:
            return len(self.processed_files)
        else:
            return len(self.raw_files)

    def get(self, idx):
        if self.new_preproc:
            return torch.load(self.processed_files[idx], weights_only=False)
        else:
            raw_data = torch.load(self.raw_files[idx], weights_only=False)

            raw_x = raw_data.x.long()
            final_x = raw_x

            if raw_data.edge_attr is not None and raw_data.edge_attr.numel() > 0:
                raw_edge_attr = raw_data.edge_attr.long()
                final_edge_attr = raw_edge_attr
            else:
                final_edge_attr = torch.zeros((0, 5), dtype=torch.long)

            final_y = raw_data.y.long()

            return Data(x=final_x, edge_index=raw_data.edge_index, edge_attr=final_edge_attr, y=final_y)


class TransactionDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        super().__init__(cfg)
        self.prepare_data()

    def prepare_data(self) -> None:
        try:
            # This fails when running outside of hydra
            from hydra.utils import get_original_cwd
            abs_datadir = os.path.join(get_original_cwd(), self.datadir)
        except (ValueError, ImportError):
            abs_datadir = self.datadir

        self.train_dataset = TransactionDataset(split='train', root=abs_datadir, datadir=abs_datadir)
        self.val_dataset = TransactionDataset(split='val', root=abs_datadir, datadir=abs_datadir)
        self.test_dataset = TransactionDataset(split='test', root=abs_datadir, datadir=abs_datadir)
        datasets = {
            'train': self.train_dataset,
            'val': self.val_dataset,
            'test': self.test_dataset
        }
        super().prepare_data(datasets)


class TransactionDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg):
        self.datamodule = datamodule
        
        self.new_preproc = False
        try:
            # This fails when running outside of hydra
            from hydra.utils import get_original_cwd
            new_encoder_path = os.path.join(get_original_cwd(), 'data', 'encoders', 'encoders.pkl')
        except (ValueError, ImportError):
            new_encoder_path = os.path.join('data', 'encoders', 'encoders.pkl')
            
        if os.path.exists(new_encoder_path):
            self.new_preproc = True
            with open(new_encoder_path, 'rb') as f:
                self.encoders = pickle.load(f)
            
            self.num_node_features = {
                'merchant_id': len(self.encoders['merchant_id_map']),
                'amt_bin': self.encoders['n_amt_bins'],
                'merchant_cluster': self.encoders['merchant_meta']['kmeans'].n_clusters + 1,
                'category_small_id': len(self.encoders['top_cats']) + 1,
                'city_id': len(self.encoders['city']),
                'time_bucket': 24 // self.encoders['time_bucket_hours']
            }
            self.num_edge_features = {
                'is_sequential': 2,
                'same_merchant_cluster': 2,
                'time_delta_bin': 10,
                'is_close_geo': 2,
            }
            self.num_y_features = {
                'gender': len(self.encoders['gender']),
                'city': len(self.encoders['city']),
                'job': len(self.encoders['job']),
                'age_bucket': len(self.encoders['age_bucket'])
            }
            self.y_feature_dims = list(self.num_y_features.values())
        else:
            datadir = datamodule.datadir
            try:
                from hydra.utils import get_original_cwd
                abs_datadir = os.path.join(get_original_cwd(), datadir)
            except (ValueError, ImportError):
                abs_datadir = datadir
            with open(os.path.join(abs_datadir, 'encoders.pkl'), 'rb') as f:
                self.encoders = pickle.load(f)
            with open(os.path.join(abs_datadir, 'bin_info.pkl'), 'rb') as f:
                self.bin_info = pickle.load(f)

            self.num_node_features = {
                'amt_bin': self.bin_info['amt'].shape[0] - 1,
                'merch_id': len(self.encoders['merchant'].classes_),
                'cat_id': len(self.encoders['category'].classes_),
                'city_pop_bin': self.bin_info['city_pop'].shape[0] - 1,
                'merch_lat_bin': self.bin_info['merch_lat'].shape[0] - 1,
                'merch_lon_bin': self.bin_info['merch_long'].shape[0] - 1,
                'unix_time_bin': self.bin_info['unix_time'].shape[0] - 1
            }
            self.num_edge_features = {
                'is_sequential': 2,
                'is_shared_attribute': 2,
                'time_delta_bin': self.bin_info['time_delta_bins'],
                'geo_distance_bin': self.bin_info['geo_distance_bins'],
                'amount_delta_bin': self.bin_info['amount_delta_bins']
            }
            self.num_y_features = {
                'gender': len(self.encoders['person_gender'].classes_),
                'city': len(self.encoders['person_city'].classes_),
                'state': len(self.encoders['person_state'].classes_),
                'job': len(self.encoders['person_job'].classes_),
                'age': self.bin_info['person_age'].shape[0] - 1
            }
        
        self.node_feature_dims = list(self.num_node_features.values())
        self.edge_feature_dims = list(self.num_edge_features.values())
        self.y_feature_dims = list(self.num_y_features.values())
        
        self.input_dims = None
        self.output_dims = None

        n_nodes = datamodule.node_counts()
        self.n_nodes = n_nodes
        self.num_classes = sum(self.node_feature_dims)
        self.max_n_nodes = len(n_nodes) - 1
        self.nodes_dist = DistributionNodes(n_nodes)

        self.node_types = datamodule.node_types()
        self.edge_types = datamodule.edge_counts()

    def compute_input_output_dims(self, datamodule, extra_features, domain_features):
        example_batch = next(iter(datamodule.train_dataloader()))

        x_emb_dim = datamodule.cfg.model.hidden_dims.x_emb_dim
        e_emb_dim = datamodule.cfg.model.hidden_dims.e_emb_dim
        y_emb_dim = datamodule.cfg.model.hidden_dims.y_emb_dim

        if self.new_preproc:
            num_node_fields = 6
            num_edge_fields = 4
            num_y_fields = 4
            
            num_continuous_node = 0
            num_continuous_edge = 0
            num_continuous_y = 0
        else:
            num_node_fields = 7
            num_edge_fields = 5
            num_y_fields = 5
            num_continuous_node = 0
            num_continuous_edge = 0
            num_continuous_y = 0


        self.input_dims = {
            'X': num_node_fields * x_emb_dim + num_continuous_node,
            'E': num_edge_fields * e_emb_dim + num_continuous_edge,
            'y': num_y_fields * y_emb_dim + num_continuous_y
        }

        ex_dense, node_mask = utils.to_dense(example_batch.x, example_batch.edge_index, example_batch.edge_attr,
                                             example_batch.batch)
        
        # Create a dummy noisy_data object to compute extra features dimensions
        noisy_data = {
            'X_t': ex_dense.X,
            'E_t': ex_dense.E,
            'y_t': example_batch.y,
            'node_mask': node_mask,
            't': torch.zeros(1, 1)
        }
        
        ex_extra_feat = extra_features(noisy_data)
        self.input_dims['X'] += ex_extra_feat.X.size(-1)
        self.input_dims['E'] += ex_extra_feat.E.size(-1)
        self.input_dims['y'] += ex_extra_feat.y.size(-1)

        ex_extra_molecular_feat = domain_features(noisy_data)
        self.input_dims['X'] += ex_extra_molecular_feat.X.size(-1)
        self.input_dims['E'] += ex_extra_molecular_feat.E.size(-1)
        self.input_dims['y'] += ex_extra_molecular_feat.y.size(-1)

        self.output_dims = {
            'X': sum(self.node_feature_dims),
            'E': sum(self.edge_feature_dims),
            'y': sum(self.y_feature_dims)
        }
