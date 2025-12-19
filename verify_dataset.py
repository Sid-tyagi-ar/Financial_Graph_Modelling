
from src.datasets.transaction_dataset import TransactionDataset, TransactionDataModule, TransactionDatasetInfos
from torch_geometric.loader import DataLoader
import pickle
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(cfg: DictConfig):
    datamodule = TransactionDataModule(cfg)
    
    train_dataset = datamodule.train_dataset
    
    print(f"Dataset size: {len(train_dataset)}")

    # Manually set node_feature_names and node_feature_dims
    train_dataset.node_feature_names = list(train_dataset.dims_node_features.keys())
    train_dataset.node_feature_dims = list(train_dataset.dims_node_features.values())

    print("Node feature names:", train_dataset.node_feature_names)
    print("Node feature dims:", train_dataset.node_feature_dims)

    dl = DataLoader(train_dataset, batch_size=1)

    for batch in dl:
        x = batch.x
        print("Batch x shape:", x.shape)
        for i, (name, dim) in enumerate(zip(train_dataset.node_feature_names, train_dataset.node_feature_dims)):
            max_val = int(x[..., i].max())
            print(f"NODE FIELD {name}: max={max_val}  allowed={dim}")
            assert max_val < dim, f"Node index out of range in {name}"
        break
    print("Dataset verification successful!")

if __name__ == '__main__':
    main()
