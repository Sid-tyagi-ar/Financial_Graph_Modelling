# Conditional Transaction Graph Generation for Fraud Detection

This project utilizes a modified version of the **DiGress** model, a discrete denoising diffusion model, to generate realistic, conditional transaction graphs. The primary application is in the domain of financial fraud detection, where generating synthetic data can help improve the robustness of fraud detection models.

The project includes a Streamlit web application that allows for interactive, conditional generation of transaction graphs based on demographic features like age, gender, and location.

## Features

*   **Conditional Graph Generation:** Generate transaction graphs based on specific user-defined conditions (e.g., age, location, job).
*   **Interactive Dashboard:** A Streamlit application provides a user-friendly interface to generate and visualize the graphs.
*   **Data Preprocessing:** A complete pipeline for processing raw transaction data into a graph-based format suitable for the DiGress model.
*   **Extensible Framework:** Built on PyTorch Lightning and Hydra, making it easy to configure and extend.

## Project Structure

```
.
├── app.py                    # The Streamlit web application
├── configs/                  # Hydra configuration files for experiments
├── preprocess_transactions.py # Script for data preprocessing
├── run_generation_script.py  # Script for generating graphs from a checkpoint
├── requirements.txt          # Python dependencies
├── src/                      # Core source code for the DiGress model and datasets
├── encoders.pkl              # Encoders for categorical features
└── bin_info.pkl              # Binning information for continuous features
```

## Pre-trained Model

The pre-trained model checkpoint (`epoch=64.ckpt`) is required to run the Streamlit application. You can download it from the following link:

[**Download Pre-trained Model (epoch=64.ckpt)**](https://your-drive-link-here.com)

Place the downloaded `epoch=64.ckpt` file in the root directory of the project.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create a Conda environment:**
    ```bash
    conda create -c conda-forge -n digress-fraud python=3.9
    conda activate digress-fraud
    ```

3.  **Install PyTorch and PyTorch Geometric:**
    Follow the official instructions for your specific hardware:
    *   [PyTorch](https://pytorch.org/)
    *   [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

4.  **Install other dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Data Preprocessing

Before training the model or running the application, you need to process your raw transaction data.

1.  **Prepare your data:** Ensure your transaction data is in a CSV file with columns like `person_id`, `amount`, `merchant`, `category`, `timestamp`, etc. A sample file `transactions.csv` is provided.

2.  **Run the preprocessing script:**
    ```bash
    python preprocess_transactions.py --csv_path transactions.csv --output_root . --n_bins 10 --fraud_only
    ```
    This will create:
    *   `processed_graphs/`: A directory with the processed graphs in `.pt` format.
    *   `encoders.pkl`: Pickled encoders for categorical features.
    *   `bin_info.pkl`: Pickled binning information for continuous features.

### 2. Training the Model

To train the DiGress model on your processed data, you can use the `main.py` script with the appropriate configuration.

```bash
python src/main.py +experiment=test
```

The `+experiment=test` argument uses the configuration defined in `configs/experiment/test.yaml`. You can create your own configuration files for different experiments.

### 3. Running the Streamlit Application

The Streamlit application uses a pre-trained model (`epoch=64.ckpt`) to generate and visualize transaction graphs.

To run the app, execute the following command:

```bash
streamlit run app.py
```

This will open a new tab in your browser with the interactive dashboard. You can then select demographic features and generate new transaction graphs.

## Citation

This work is based on the DiGress model. If you use this code in your research, please cite the original paper:

```
@article{vignac2022digress,
  title={DiGress: Discrete Denoising diffusion for graph generation},
  author={Vignac, Clement and Krawczuk, Igor and Siraudin, Antoine and Wang, Bohan and Cevher, Volkan and Frossard, Pascal},
  journal={arXiv preprint arXiv:2209.14734},
  year={2022}
}
```