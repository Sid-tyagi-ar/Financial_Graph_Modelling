import torch
import pickle
import os
import numpy as np
from omegaconf import OmegaConf
import torch.nn.functional as F
import argparse

# Import project-specific classes
from src.diffusion_model_discrete import DiscreteDenoisingDiffusion
from src.datasets.transaction_dataset import TransactionDatasetInfos, TransactionDataModule
from torch_geometric.data import Data

# This patch is still needed for manual loading if PyTorch version is high
import torch.serialization
_original_torch_load = torch.load
def _patched_torch_load(f, map_location=None, pickle_module=pickle, *, weights_only=None, mmap=None, **kwargs):
    return _original_torch_load(f, map_location=map_location, pickle_module=pickle_module, weights_only=False, mmap=mmap, **kwargs)
torch.load = _patched_torch_load
import pytorch_lightning as pl


def load_model_and_generate(args):
    """
    Loads the model and generates a single graph.
    """
    print("--- Loading model and data ---")
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint file not found at {checkpoint_path}")
        return

    try:
        # --- Step 1: Load the entire checkpoint and extract config ---
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        cfg = OmegaConf.create(ckpt['hyper_parameters']['cfg'])
        print("Successfully loaded configuration from checkpoint.")

        # --- Step 2: Build all dependencies from the loaded config ---
        cfg.dataset.datadir = "."
        cfg.train.batch_size = 1
        cfg.train.num_workers = 0

        with open('encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        with open('bin_info.pkl', 'rb') as f:
            bin_info = pickle.load(f)

        datamodule = TransactionDataModule(cfg)
        dataset_infos = TransactionDatasetInfos(datamodule, cfg)

        from src.diffusion.extra_features import ExtraFeatures, DummyExtraFeatures
        if cfg.model.get('extra_features') is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule, extra_features, domain_features)

        # --- Step 3: Manually create a new model instance ---
        print("Manually creating new model instance...")
        model = DiscreteDenoisingDiffusion(
            cfg=cfg, dataset_infos=dataset_infos, train_metrics=None,
            sampling_metrics=None, visualization_tools=None,
            extra_features=extra_features, domain_features=domain_features
        )

        # --- Step 4: Manually load the weights ---
        if 'ema_state_dict' in ckpt and ckpt['ema_state_dict']:
            print("Applying Exponential Moving Average (EMA) weights...")
            model.load_state_dict(ckpt['ema_state_dict'], strict=False)
        else:
            print("Applying standard model weights...")
            model.load_state_dict(ckpt['state_dict'], strict=False)
        
        print("Weights loaded successfully!")
        model.eval() 
        
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        import traceback
        print(traceback.format_exc())
        return

    print("\n--- Generating Graph ---")
    try:
        # 1. Encode demographics from args
        gender_id = encoders['person_gender'].transform([args.gender])[0]
        city_id = encoders['person_city'].transform([args.city])[0]
        state_id = encoders['person_state'].transform([args.state])[0]
        job_id = encoders['person_job'].transform([args.job])[0]
        
        age_bins = bin_info['person_age']
        age_id = np.digitize(args.age, bins=age_bins) - 1
        age_id = np.clip(age_id, 0, len(age_bins) - 2)

        y_features = [gender_id, city_id, state_id, job_id, age_id]
        y_tensor_int = torch.tensor([y_features], dtype=torch.long)

        y_encoded = []
        for i, dim in enumerate(model.dataset_info.y_feature_dims):
            y_encoded.append(F.one_hot(y_tensor_int[:, i], num_classes=dim))
        y_final = torch.cat(y_encoded, dim=-1).float()
        print("Successfully encoded demographic data.")

        # 2. Set diffusion steps
        model.T = args.diffusion_steps

        # 3. Generate graph
        print(f"Calling sample_batch with {args.num_nodes} nodes and {args.diffusion_steps} steps...")
        
        generated_graphs = model.sample_batch(
            batch_id=0, batch_size=1, keep_chain=0, number_chain_steps=0,
            save_final=0, num_nodes=torch.tensor([args.num_nodes]), y=y_final
        )
        
        # 4. Save the output
        X_one_hot, E_one_hot = generated_graphs[0]
        output_data = {'X': X_one_hot, 'E': E_one_hot}
        torch.save(output_data, args.output_file)
        print(f"\n✅ Graph successfully generated and saved to '{args.output_file}'")

    except Exception as e:
        print(f"An error occurred during generation: {e}")
        import traceback
        print(traceback.format_exc())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a single graph using a trained DiGress model.')
    parser.add_argument('--checkpoint', type=str, default='epoch=64.ckpt', help='Path to model checkpoint.')
    parser.add_argument('--output_file', type=str, default='output_graph.pt', help='Path to save the generated graph.')
    
    # Demographic arguments with placeholder defaults
    parser.add_argument('--gender', type=str, default='DEFAULT', help='Gender (e.g., M, F)')
    parser.add_argument('--city', type=str, default='DEFAULT', help='City name')
    parser.add_argument('--state', type=str, default='DEFAULT', help='State abbreviation')
    parser.add_argument('--job', type=str, default='DEFAULT', help='Job title')
    parser.add_argument('--age', type=int, default=45, help='Age')
    
    # Generation parameters
    parser.add_argument('--num_nodes', type=int, default=15, help='Number of transactions (nodes) to generate.')
    parser.add_argument('--diffusion_steps', type=int, default=250, help='Number of inference diffusion steps.')

    cli_args = parser.parse_args()

    # --- Dynamically set valid defaults if user didn't provide them ---
    try:
        with open('encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        
        if cli_args.gender == 'DEFAULT':
            cli_args.gender = encoders['person_gender'].classes_[0]
        if cli_args.city == 'DEFAULT':
            city_index = min(10, len(encoders['person_city'].classes_) - 1)
            cli_args.city = encoders['person_city'].classes_[city_index]
        if cli_args.state == 'DEFAULT':
            state_index = min(5, len(encoders['person_state'].classes_) - 1)
            cli_args.state = encoders['person_state'].classes_[state_index]
        if cli_args.job == 'DEFAULT':
            job_index = min(20, len(encoders['person_job'].classes_) - 1)
            cli_args.job = encoders['person_job'].classes_[job_index]
    except FileNotFoundError:
        print("Warning: encoders.pkl not found. Using hardcoded defaults. This may fail.")
        if cli_args.gender == 'DEFAULT': cli_args.gender = 'M'
        if cli_args.city == 'DEFAULT': cli_args.city = 'New York'
        if cli_args.state == 'DEFAULT': cli_args.state = 'NY'
        if cli_args.job == 'DEFAULT': cli_args.job = 'Architect'

    load_model_and_generate(cli_args)
