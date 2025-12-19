
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import os
from hydra.utils import get_original_cwd
import torch.nn.functional as F

# Add the project root to the Python path to allow imports from src
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.transaction_dataset import TransactionDataModule, TransactionDatasetInfos
from src.diffusion.extra_features import ExtraFeatures, DummyExtraFeatures
from src.metrics.train_metrics import TrainLossDiscrete
from src import utils

@hydra.main(config_path="configs", config_name="config", version_base="1.1")
def debug_pipeline(cfg: DictConfig):
    """
    This script isolates the data pipeline and loss calculation to debug tensor shapes.
    """
    # Manually load and merge the experiment and dataset configs
    exp_config = OmegaConf.load(os.path.join(get_original_cwd(), "configs/experiment/test.yaml"))
    data_config = OmegaConf.load(os.path.join(get_original_cwd(), "configs/dataset/transaction.yaml"))
    cfg = OmegaConf.merge(cfg, exp_config)
    cfg.dataset = OmegaConf.merge(cfg.dataset, data_config)

    print("--- Initializing DataModule and Features ---")
    datamodule = TransactionDataModule(cfg)
    datamodule.prepare_data()
    dataset_infos = TransactionDatasetInfos(datamodule, cfg)

    if cfg.model.extra_features is not None:
        extra_features = ExtraFeatures(cfg.model.extra_features, dataset_infos)
    else:
        extra_features = DummyExtraFeatures()
    domain_features = DummyExtraFeatures()

    dataset_infos.compute_input_output_dims(
        datamodule=datamodule, extra_features=extra_features, domain_features=domain_features
    )

    print("--- Getting a single batch ---")
    batch = next(iter(datamodule.train_dataloader()))

    print("--- Mimicking training_step ---")
    dense_data, node_mask = utils.to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    dense_data = dense_data.mask(node_mask)
    true_X, true_E = dense_data.X, dense_data.E

    # Dummy prediction tensor with the correct shape
    pred_X = torch.randn_like(true_X)
    pred_E = torch.randn_like(true_E)
    pred_y = torch.randn_like(batch.y)

    print(f"Shape of true_X before loss: {true_X.shape}")
    print(f"Shape of pred_X before loss: {pred_X.shape}")

    print("\n--- Mimicking TrainLossDiscrete.forward ---")
    
    # Reshape
    true_X_reshaped = torch.reshape(true_X, (-1, true_X.size(-1)))
    pred_X_reshaped = torch.reshape(pred_X, (-1, pred_X.size(-1)))
    print(f"Shape of true_X after reshape: {true_X_reshaped.shape}")
    print(f"Shape of pred_X after reshape: {pred_X_reshaped.shape}")

    # Mask
    mask_X = (true_X_reshaped != 0.).any(dim=-1)
    print(f"Number of True values in mask_X: {mask_X.sum()}")

    flat_true_X = true_X_reshaped[mask_X, :]
    flat_pred_X = pred_X_reshaped[mask_X, :]
    print(f"Shape of flat_true_X after mask: {flat_true_X.shape}")
    print(f"Shape of flat_pred_X after mask: {flat_pred_X.shape}")

    # Feature-wise loss calculation
    node_feature_dims = list(dataset_infos.num_node_features.values())
    current_dim = 0
    for i, feature_dim in enumerate(node_feature_dims):
        print(f"\n--- Feature {i} (dim={feature_dim}) ---")
        pred_slice = flat_pred_X[:, current_dim : current_dim + feature_dim]
        true_slice = flat_true_X[:, current_dim : current_dim + feature_dim]
        
        print(f"Shape of pred_slice: {pred_slice.shape}")
        print(f"Shape of true_slice: {true_slice.shape}")

        if true_slice.numel() == 0:
            print("True slice is empty, skipping.")
            current_dim += feature_dim
            continue

        true_indices = torch.argmax(true_slice, dim=-1)
        print(f"Shape of true_indices: {true_indices.shape}")

        try:
            # Dummy loss calculation to check shapes
            F.cross_entropy(pred_slice, true_indices)
            print("CrossEntropy check: SUCCESS")
        except ValueError as e:
            print(f"CrossEntropy check: FAILED")
            print(f"  ERROR: {e}")

        current_dim += feature_dim

    print("\n--- Analyzing edge type distribution in training data ---")
    edge_types = torch.argmax(dense_data.E, dim=-1)
    unique_edge_types, counts = torch.unique(edge_types, return_counts=True)
    for et, count in zip(unique_edge_types, counts):
        print(f"Edge type {et.item()}: {count.item()} occurrences")

if __name__ == "__main__":
    debug_pipeline()
