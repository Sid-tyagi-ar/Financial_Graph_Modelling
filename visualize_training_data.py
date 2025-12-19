import os
import torch
from glob import glob
from omegaconf import OmegaConf

from src.analysis.visualization import NonMolecularVisualization
from src.datasets.transaction_dataset import TransactionDatasetInfos, TransactionDataModule


def visualize_training_graphs(num_to_visualize=5):
    """Loads graphs from the training set and saves their visualizations as PNGs."""
    print(f"--- Visualizing {num_to_visualize} graphs from the training set ---")

    # 1. Setup paths and create output directory
    output_dir = 'training_graph_samples'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving images to: {output_dir}/")

    # 2. Load dataset metadata required by the visualizer for decoding features
    try:
        print("Loading dataset info...")
        # Create a dummy config to load the dataset module
        dummy_cfg = OmegaConf.create({
            "general": {"name": "visualization"},
            "dataset": {"datadir": "."}, 
            "train": {"batch_size": 1, "num_workers": 0}
        })
        datamodule = TransactionDataModule(dummy_cfg)
        dataset_infos = TransactionDatasetInfos(datamodule, dummy_cfg)
        visualizer = NonMolecularVisualization(dataset_infos=dataset_infos)
        train_dataset = datamodule.train_dataloader().dataset
    except FileNotFoundError as e:
        print(f"Error: Could not load dataset files. {e}")
        print("Please ensure 'encoders.pkl', 'bin_info.pkl', and the 'processed_graphs' directory are present.")
        return

    if len(train_dataset) == 0:
        print("Training dataset is empty. Cannot visualize.")
        return

    # 3. Loop through the first few graphs, load them, and visualize
    num_to_visualize = min(num_to_visualize, len(train_dataset))
    for i in range(num_to_visualize):
        print(f"Processing graph {i}...")
        
        # Get the data object with one-hot encoded features from the dataset
        data_obj_ohe = train_dataset.get(i)

        # Convert to a networkx graph for visualization
        nx_graph = visualizer.to_networkx(data_obj_ohe)

        # Save the visualization
        output_path = os.path.join(output_dir, f"training_graph_sample_{i}.png")
        visualizer.visualize_non_molecule(graph=nx_graph, pos=None, path=output_path)
        print(f"  Saved visualization to {output_path}")

    print("\nVisualization of training graphs complete.")

if __name__ == '__main__':
    visualize_training_graphs()
