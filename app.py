import streamlit as st
import torch
import pickle
import os
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
import subprocess

# Import project-specific classes
from src.diffusion_model_discrete import DiscreteDenoisingDiffusion
from src.datasets.transaction_dataset import TransactionDatasetInfos, TransactionDataModule
from src.analysis.visualization import NonMolecularVisualization
from torch_geometric.data import Data

# CRITICAL FIX: Monkey-patch torch.load itself before Lightning imports it
import torch.serialization
_original_torch_load = torch.load

def _patched_torch_load(f, map_location=None, pickle_module=pickle, *, weights_only=None, mmap=None, **kwargs):
    """Force weights_only=False for all torch.load calls"""
    return _original_torch_load(
        f, 
        map_location=map_location, 
        pickle_module=pickle_module,
        weights_only=False,  # Always False
        mmap=mmap,
        **kwargs
    )

# Apply the patch globally
torch.load = _patched_torch_load

# Now import Lightning (after patching torch.load)
import pytorch_lightning as pl


# --- 1. Load Model and Data (Cached) ---
@st.cache_resource
def load_model_and_data():
    """
    Loads the model from checkpoint MANUALLY to avoid class loading issues.
    """
    st.write("Cache miss: Loading model and data...")
    
    checkpoint_path = "epoch=64.ckpt" 
    if not os.path.exists(checkpoint_path):
        st.error(f"Checkpoint file not found at {checkpoint_path}. Please place it in the root directory.")
        return None, None, None, None

    try:
        # --- Step 1: Load the entire checkpoint and extract config ---
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        cfg = OmegaConf.create(ckpt['hyper_parameters']['cfg'])
        st.info("Successfully loaded configuration from checkpoint.")

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
        # We pass dummy metrics/viz tools because they are not needed for inference.
        model = DiscreteDenoisingDiffusion(
            cfg=cfg,
            dataset_infos=dataset_infos,
            train_metrics=None,
            sampling_metrics=None,
            visualization_tools=None,
            extra_features=extra_features,
            domain_features=domain_features
        )

        # --- Step 4: Manually load the weights ---
        if 'ema_state_dict' in ckpt and ckpt['ema_state_dict']:
            st.info("Applying Exponential Moving Average (EMA) weights...")
            model.load_state_dict(ckpt['ema_state_dict'], strict=False)
        else:
            st.info("Applying standard model weights...")
            model.load_state_dict(ckpt['state_dict'], strict=False)
        
        st.success("✅ Weights loaded successfully!")
        model.eval()
        
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        st.error("Full error details:")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None, None
    
    visualizer = NonMolecularVisualization(dataset_infos)

    st.write("✅ All components loaded successfully.")
    return model, encoders, bin_info, visualizer


# --- Helper Functions for Decoding and Visualization ---

def decode_graph_to_dataframe(graph_data, encoders, bin_info):
    """Decodes the raw graph data tensor into a human-readable DataFrame."""
    decoded_rows = []
    node_features = graph_data.x.numpy()
    
    feature_map = [
        ('Amount', 'amt', 'bin'),
        ('Merchant', 'merchant', 'encoder'),
        ('Category', 'category', 'encoder'),
        ('City Population', 'city_pop', 'bin'),
        ('Merchant Latitude', 'merch_lat', 'bin'),
        ('Merchant Longitude', 'merch_long', 'bin'),
        ('Timestamp', 'unix_time', 'bin')
    ]

    for i in range(graph_data.num_nodes):
        row = {'Transaction #': i}
        for j, (feat_name, key, dec_type) in enumerate(feature_map):
            val_id = node_features[i, j]
            try:
                if dec_type == 'encoder':
                    row[feat_name] = encoders[key].inverse_transform([val_id])[0]
                elif dec_type == 'bin':
                    bins = bin_info[key]
                    lower_bound = bins[val_id]
                    upper_bound = bins[val_id + 1]
                    row[feat_name] = f"[{lower_bound:.2f}, {upper_bound:.2f}]"
            except (IndexError, KeyError):
                row[feat_name] = "N/A"
        decoded_rows.append(row)
        
    return pd.DataFrame(decoded_rows).set_index('Transaction #')

def create_graph_visualization(graph_data, df_display):
    """Creates a networkx visualization of the generated graph."""
    G = nx.Graph()
    
    for i in range(graph_data.num_nodes):
        G.add_node(i, label=f"Txn {i}\n{df_display.loc[i, 'Merchant']}\n{df_display.loc[i, 'Amount']}")

    edge_attrs = graph_data.edge_attr.numpy()
    edge_labels = {}
    edge_colors = []
    
    for i in range(graph_data.edge_index.shape[1]):
        u, v = graph_data.edge_index[0, i].item(), graph_data.edge_index[1, i].item()
        if u >= v: continue

        attr = edge_attrs[i]
        is_sequential = attr[0] == 1
        is_shared = attr[1] == 1
        
        label = []
        color = 'gray'
        if is_sequential:
            label.append("Sequential")
            color = 'orange'
        if is_shared:
            label.append("Shared Attr")
            color = 'blue' if color == 'gray' else 'red'
            
        G.add_edge(u, v)
        edge_labels[(u, v)] = ", ".join(label)
        edge_colors.append(color)

    fig, ax = plt.subplots(figsize=(16, 12))
    pos = nx.spring_layout(G, seed=42, k=1.2/np.sqrt(G.number_of_nodes()))
    
    node_labels = nx.get_node_attributes(G, 'label')
    
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=2000, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=1.5, ax=ax, connectionstyle='arc3,rad=0.1')
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='black', ax=ax)
    
    ax.set_title("Generated Transaction Graph", fontsize=20)
    plt.tight_layout()
    return fig


# --- 2. Main App Logic ---
st.set_page_config(layout="wide")
st.title("🔍 Conditional Graph Generation for Fraud Detection")

model, encoders, bin_info, visualizer = load_model_and_data()

if model is None:
    st.stop()

# --- 3. Sidebar for User Inputs ---
st.sidebar.header("🎛️ Conditional Generation Controls")

gender = st.sidebar.selectbox("Gender", options=encoders['person_gender'].classes_, index=0)
state = st.sidebar.selectbox("State", options=encoders['person_state'].classes_, index=5)
city = st.sidebar.selectbox("City", options=encoders['person_city'].classes_, index=10)
job = st.sidebar.selectbox("Job", options=encoders['person_job'].classes_, index=20)
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=45)
num_nodes = st.sidebar.slider("Number of Transactions to Generate", min_value=5, max_value=50, value=15, step=1)
diffusion_steps = st.sidebar.slider("Inference Diffusion Steps", min_value=50, max_value=1000, value=250, step=50)
generate_button = st.sidebar.button("🎲 Generate Graph", type="primary")

# --- 4. Main Panel for Output ---
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📊 Generated Graph")
    if generate_button:
        with st.spinner("Generating graph in a separate process... This may take a moment."):
            # 1. Construct the command for the generate.py script
            output_file = "streamlit_output_graph.pt"
            command = [
                'python',
                'run_generation_script.py',
                '--gender', gender,
                '--city', city,
                '--state', state,
                '--job', job,
                '--age', str(age),
                '--num_nodes', str(num_nodes),
                '--diffusion_steps', str(diffusion_steps),
                '--output_file', output_file
            ]

            # 2. Run the script as a subprocess
            process = subprocess.run(command, capture_output=True, text=True)

            if process.returncode == 0:

                st.success("Generation script finished successfully.")

                st.subheader("Debug Info: Generation Script Output")

                st.code(process.stdout)            

                # 3. Load the result from the file

                try:

                    st.subheader("Debug Info: Graph Processing")

                    st.write("Loading graph from `streamlit_output_graph.pt`...")

                    graph_output = torch.load(output_file)

                    X_one_hot = graph_output['X']

                    E_one_hot = graph_output['E']

                    st.write(f"Loaded `X_one_hot` with shape: `{X_one_hot.shape}`")

                    st.write(f"Loaded `E_one_hot` with shape: `{E_one_hot.shape}`")
            
                    # 4. Decode and Visualize

                    st.write("Decoding one-hot tensors into integer format...")

                    node_dims = visualizer.dataset_infos.node_feature_dims
                    st.write(f"Using node feature dimensions: `{node_dims}`")
                    decoded_node_attrs = []
                    current_dim = 0
                    for dim in node_dims:
                        feature_slice = X_one_hot[:, current_dim : current_dim + dim]
                        decoded_feature = torch.argmax(feature_slice, dim=-1)
                        decoded_node_attrs.append(decoded_feature.unsqueeze(1))
                        current_dim += dim
                    X_int = torch.cat(decoded_node_attrs, dim=1)

                    st.write(f"Decoded `X_int` with shape: `{X_int.shape}`")

                    edge_mask = E_one_hot.sum(dim=-1) > 0.5

                    edge_index = torch.nonzero(edge_mask).t()

                    st.write(f"Reconstructed `edge_index` with shape: `{edge_index.shape}`")
            
                    edge_attr_one_hot = E_one_hot[edge_index[0], edge_index[1]]

                    edge_dims = visualizer.dataset_infos.edge_feature_dims

                    st.write(f"Using edge feature dimensions: `{edge_dims}` which has {len(edge_dims)} features.")

                    decoded_edge_attrs = []

                    current_dim = 0

                    for i, dim in enumerate(edge_dims):

                        feature_slice = edge_attr_one_hot[:, current_dim : current_dim + dim]

                        decoded_feature = torch.argmax(feature_slice, dim=-1)

                        unsqueezed_feature = decoded_feature.unsqueeze(1)

                        decoded_edge_attrs.append(unsqueezed_feature)

                        current_dim += dim

                    st.write(f"Number of tensors to concatenate: {len(decoded_edge_attrs)}")

                    edge_attr_int = torch.cat(decoded_edge_attrs, dim=1)

                    st.write(f"Final `edge_attr_int` shape: `{edge_attr_int.shape}`")

                    graph_data = Data(x=X_int, edge_index=edge_index, edge_attr=edge_attr_int, num_nodes=X_int.size(0))

                    st.write("Created PyG Data object:")

                    st.code(str(graph_data))

                    df_display = decode_graph_to_dataframe(graph_data, encoders, bin_info)

                    fig = create_graph_visualization(graph_data, df_display)

                    st.pyplot(fig)

                    st.subheader("Generated Transaction Details")

                    st.dataframe(df_display)

                except FileNotFoundError:
                    st.error(f"Output file '{output_file}' not found. Generation may have failed.")
                    st.subheader("Generation Script Output (stdout):")
                    st.code(process.stdout)
                    st.subheader("Generation Script Error (stderr):")
                    st.code(process.stderr)
                except Exception as e:
                    st.error(f"An error occurred while processing the generated graph: {e}")

            else:
                st.error("The generation script failed. See details below.")
                st.subheader("Generation Script Output (stdout):")
                st.code(process.stdout)
                st.subheader("Generation Script Error (stderr):")
                st.code(process.stderr)
    else:
        st.info("👈 Select demographic features and click **'Generate Graph'** to begin.")

with col2:
    st.subheader("ℹ️ Model & Generation Info")
    if model is not None:
        st.metric("Model Type", "DiGress Diffusion")
        st.metric("Training Diffusion Steps", model.hparams.cfg.model.diffusion_steps)
        
        with st.expander("📋 Available Encoders"):
            for key in encoders.keys():
                st.write(f"- `{key}`: {len(encoders[key].classes_)} classes")
        
        with st.expander("Binning Info"):
            for key, val in bin_info.items():
                if isinstance(val, np.ndarray):
                    st.write(f"- `{key}`: {len(val)-1} bins")
                else:
                    st.write(f"- `{key}`: {val} bins")