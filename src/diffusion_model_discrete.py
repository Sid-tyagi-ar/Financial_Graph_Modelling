


import torch
"""
Module docstring for DiscreteDenoisingDiffusion
This module implements a PyTorch LightningModule that trains and samples from a
discrete denoising diffusion model on graph-structured data. The implementation
wraps a transformer-based neural network (GraphTransformer) and a discrete
diffusion process defined by a noise schedule and transition model.
Primary class:
- DiscreteDenoisingDiffusion: LightningModule that provides training, validation,
    testing and sampling procedures for discrete graph diffusion models.
High-level behavior:
- During training/validation/test, input graphs (node types X, edge types E and
    optional graph-level labels y) are one-hot encoded and corrupted via a
    discrete transition model (apply_noise). The model is trained to predict the
    denoised distributions (pred.X, pred.E, pred.y) conditioned on noisy inputs.
- The training objective contains cross-entropy reconstruction terms, KL terms
    for the diffusion prior, and additional terms that implement a variational
    estimator of the ELBO for the discrete diffusion process.
- Sampling constructs graphs by starting from a discrete prior (z_T sampled
    from the limit/ uniform distribution) and iteratively sampling p(z_{t-1} |
    z_t) using model predictions (sample_batch and sample_p_zs_given_zt).
- Several metrics are tracked and logged (WandB + Lightning logger) for both
    stability and sampling evaluation.
Class: DiscreteDenoisingDiffusion
--------------------------------
Purpose:
        Train and sample from a discrete denoising diffusion model over graphs.
        It coordinates:
            - input encoding/decoding between integer fields and one-hot encodings,
            - adding and sampling noise via transition matrices and noise schedule,
            - calling the GraphTransformer for predictions,
            - computing ELBO-related losses and metrics,
            - iterative discrete sampling for generation,
            - logging and visualization integration (wandb and optional visualization_tools).
Constructor arguments (expected):
        cfg: configuration object with nested attributes used throughout (typical
                 fields: general.name, general.log_every_steps, general.sample_every_val,
                 general.samples_to_generate, general.samples_to_save, general.chains_to_save,
                 general.final_model_samples_to_generate, final_model_samples_to_save,
                 final_model_chains_to_save, model: {diffusion_steps, diffusion_noise_schedule,
                 transition, n_layers, hidden_mlp_dims, hidden_dims, lambda_train},
                 train: {lr, weight_decay, batch_size}, ...).
        dataset_infos: object/Namespace describing dataset feature sizes and shapes:
                 - input_dims: dict with X, E, y input dimensionalities (pre-one-hot)
                 - output_dims: dict with X, E, y output dimensionalities (one-hot classes)
                 - node_feature_dims: list[int] lengths for each categorical node feature field
                 - edge_feature_dims: list[int] lengths for each categorical edge field
                 - y_feature_dims: list[int] lengths for each categorical y fields
                 - node_types, edge_types: tensors used for marginal transition initialization
                 - nodes_dist: distribution object for number of nodes sampling (has sample_n and log_prob)
                 - node_types/edge_types may be tensors of counts used to create marginal priors
        train_metrics: callable/metric aggregator used during training (reset/update/log)
        sampling_metrics: callable used to evaluate generated samples
        visualization_tools: optional object providing visualize and visualize_chain methods
        extra_features: callable that computes extra input features for the network from noisy_data
        domain_features: optional dataset domain-specific features (kept for completeness)
Public attributes (selected):
        - model: GraphTransformer used to predict denoised categorical distributions
        - noise_schedule: object implementing a continuous noise schedule and alpha_bar retrieval
        - transition_model: object implementing discrete transition matrices Qt, Qt_bar, get_Qt_bar, get_Qt
        - limit_dist: a PlaceHolder object containing the uniform / marginal prior used as p(z_T)
        - train_loss, val_nll, test_nll, val_CE, val_X_kl, val_E_kl, etc.: metrics for tracking
        - T: number of diffusion timesteps
        - Xdim_output, Edim_output, ydim_output: integer number of classes for outputs
        - dataset_info: original dataset_infos object
Key methods and behavior:
        training_step(data, i)
                - Prepares dense tensors from sparse graph data (utils.to_dense).
                - One-hot encodes node, edge and y categorical fields according to
                    dataset_info.{node_feature_dims, edge_feature_dims, y_feature_dims}.
                - Calls apply_noise to sample a timestep t and produce noisy data z_t.
                - Computes extra_data = compute_extra_data(noisy_data).
                - Runs forward() to obtain model predictions (logits) and computes the
                    training loss via self.train_loss and updates train_metrics.
                - Returns {'loss': loss} (compatible with Lightning).
        validation_step(data, i)
                - Same preprocessing as training_step; computes noisy_data with apply_noise.
                - Runs forward, computes stable CE loss via train_loss and stores it in val_CE.
                - Computes the full NLL estimator and other metrics via compute_val_loss.
                - Returns {'loss': ce_loss}.
        test_step(data, i)
                - Same as validation_step but logs into test metrics and returns NLL loss.
        configure_optimizers()
                - Returns AdamW optimizer configured from cfg.train settings.
        on_fit_start(), on_train_epoch_start(), on_train_epoch_end(), on_validation_epoch_start(),
        on_validation_epoch_end(), on_test_epoch_start(), on_test_epoch_end()
                - Lifecycle hooks that reset metrics, log epoch-level metrics to WandB and
                    Lightning logger, trigger periodic sampling/visualization, save best val CE.
        apply_noise(X, E, y, node_mask)
                - Samples discrete timestep t in [0, T] (training allows t=0; eval starts at t>=1).
                - Computes transition Qt_bar for alpha_t_bar and samples z_t ~ q(z_t | z_0) by
                    applying Qt_bar to one-hot X and E features.
                - Returns noisy_data dict:
                        { 't_int', 't' (normalized), 'beta_t', 'alpha_s_bar', 'alpha_t_bar',
                            'X_t', 'E_t', 'y_t', 'node_mask' }
                - Shapes:
                        X, X_t: (bs, n, dx_out),
                        E, E_t: (bs, n, n, de_out),
                        y_t: (bs, sum(y_feature_dims)) as concatenated one-hot,
                        node_mask: boolean (bs, n).
        forward(noisy_data, extra_data, node_mask)
                - Converts one-hot noisy fields X_t, E_t, y_t into integer fields per categorical
                    sub-feature (splits by node_feature_dims, edge_feature_dims, y_feature_dims).
                - Embeds integer fields using model.embed_X/embed_E/embed_y and calls the GraphTransformer.
                - Returns the model output object with logits for X, E, y (not softmaxed).
        kl_prior(X, E, y, node_mask)
                - Computes KL between q(z_T | x) and the chosen prior p(z_T) (limit_dist).
                - Uses the transition_model to compute Qt_bar(T) and computes probX = X @ Qtb.X etc.
                - Masks invalid entries according to node_mask and sums the KL over non-batch dims.
                - Returns per-example KL values summed across features (tensor of shape (bs,)).
        compute_Lt(X, E, y, pred, noisy_data, node_mask, test)
                - Constructs posterior distributions for q(z_{t-1} | z_t, x) (via diffusion_utils.posterior_distributions)
                    for both the true posterior (using true X,E,y) and predicted posterior (using model softmax outputs).
                - Masks and renormalizes distributions with small eps for numerical stability.
                - Computes KLs for node and edge posteriors and updates val/test metrics.
                - Returns a scalar tensor or per-example loss summed appropriately.
        reconstruction_logp(t, X, E, y, node_mask)
                - Implements the L0 / reconstruction term: -log p(x | z_0), computed by sampling z_0 given X,E
                    through Q0 = transition_model.get_Qt(beta_0) and using the model to predict p(x|z_0).
                - Returns predicted categorical probabilities (PlaceHolder with X, E, y probabilities).
        compute_val_loss(pred, noisy_data, X, E, y, node_mask, test=False)
                - Computes the per-example variational lower-bound estimator combining:
                        - negative log prior over number of nodes: -log p_N
                        - KL between q(z_T | x) and p(z_T) (kl_prior)
                        - diffusion loss terms computed by compute_Lt
                        - reconstruction term computed with reconstruction_logp
                - Updates val/test NLL metric and logs components to WandB.
                - Returns or logs the aggregated batch NLL (metric object handles averaging).
        sample_batch(batch_id, batch_size, keep_chain, number_chain_steps, save_final, num_nodes=None, y=None)
                - Generates batch_size graphs by sampling from p(z_{t-1} | z_t) iteratively from T -> 0.
                - When y is provided (integer fields), it is one-hot encoded and fixed during sampling.
                - Optionally stores intermediate chains (for visualization) and final generated graphs.
                - Saves generated graphs to disk (graphs/<name>/...) if save_final > 0.
                - Returns a list of generated graphs (one-hot node and edge tensors) for analysis.
        sample_p_zs_given_zt(t, X_t, E_t, y_t, node_mask, last_step: bool)
                - Single-step sampler for p(z_{s} | z_{t}) used by sample_batch.
                - Calls forward() to obtain model logits for the current z_t, computes posterior mixture
                    probabilities and samples discrete z_s accordingly.
                - If last_step is True, also returns a sampled predicted_graph produced directly from model
                    softmax (useful for visualizing network predictions on final step).
        compute_extra_data(noisy_data)
                - Delegates to the provided extra_features callable to compute auxiliary inputs for the model.
Notes, expected types and shapes:
        - All tensors are assumed to be torch tensors and on the correct device (module uses self.device).
        - Many functions use one-hot representations as float tensors for inputs to transition matrices.
        - dataset_info.{node_feature_dims, edge_feature_dims, y_feature_dims} describe how to split
            concatenated one-hot vectors into individual categorical features.
        - The outputs of model.forward are logits that must be softmaxed before being interpreted as probabilities.
        - Numerical stability: small epsilons are added before log/normalization in several places.
Exceptions:
        - Raises ValueError if y feature values exceed declared number of classes (out-of-bounds).
        - Asserts enforce distribution normalization and symmetry of edge types where required.
Integration and side effects:
        - Uses WandB for logging various scalar metrics; Lightning logger is used for checkpointing metric 'val_loss'.
        - Optionally uses visualization_tools to write visualization outputs to disk.
        - Saves generated graphs to disk under 'graphs/<name>/...' during sampling.
External dependencies:
        - src.models.transformer_model.GraphTransformer
        - src.diffusion.noise_schedule (PredefinedNoiseScheduleDiscrete et al.)
        - src.diffusion.transition models (DiscreteUniformTransition, BlockDiagonalDiscreteUniformTransition, MarginalUniformTransition)
        - src.diffusion.diffusion_utils: utility functions for discrete posterior computation, sampling and masking
        - src.metrics.*: metric classes used for tracking training/validation/test losses
        - src.utils: helpers such as PlaceHolder, to_dense conversions and masking utilities
        - torch, pytorch_lightning, torchmetrics, wandb, os, time
This docstring documents the intended behavior, inputs/outputs and side effects of the
DiscreteDenoisingDiffusion LightningModule implemented in this file. For implementation
details and precise tensor shapes for every operation, refer to the function docstrings
and the helper modules imported above (diffusion_utils, transition_model, noise_schedule).
"""
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import time
import wandb
import os

from src.models.transformer_model import GraphTransformer
from src.diffusion.noise_schedule import DiscreteUniformTransition, PredefinedNoiseScheduleDiscrete,MarginalUniformTransition, BlockDiagonalDiscreteUniformTransition
from src.diffusion import diffusion_utils
from src.metrics.train_metrics import TrainLossDiscrete
from src.metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL
from src import utils
from torchmetrics import MeanMetric


class DiscreteDenoisingDiffusion(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, train_metrics, sampling_metrics, visualization_tools, extra_features,
                 domain_features):
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.cfg = cfg
        self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.T = cfg.model.diffusion_steps

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.y_feature_dims = dataset_infos.y_feature_dims
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos

        self.train_loss = TrainLossDiscrete(self.cfg.model.lambda_train, dataset_infos=self.dataset_info)

        self.val_nll = NLL()
        self.val_CE = MeanMetric()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_y_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()
        self.val_y_logp = SumExceptBatchMetric()

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_y_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()
        self.test_y_logp = SumExceptBatchMetric()

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        self.model = GraphTransformer(n_layers=cfg.model.n_layers,
                                      input_dims=input_dims,
                                      hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                      hidden_dims=cfg.model.hidden_dims,
                                      output_dims=output_dims,
                                      dataset_infos=dataset_infos,
                                      act_fn_in=nn.ReLU(),
                                      act_fn_out=nn.ReLU())

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule,
                                                              timesteps=cfg.model.diffusion_steps)

        if cfg.model.transition == 'uniform':
            self.transition_model = BlockDiagonalDiscreteUniformTransition(
                x_classes=self.Xdim_output, e_classes=self.Edim_output, y_feature_dims=self.y_feature_dims)
            x_limit = torch.ones(self.Xdim_output) / self.Xdim_output
            e_limit = torch.ones(self.Edim_output) / self.Edim_output
            y_limit = torch.ones(self.ydim_output) / self.ydim_output
            self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)

        elif cfg.model.transition == 'marginal':
            node_types = self.dataset_info.node_types.float()
            x_marginals = node_types / torch.sum(node_types)

            edge_types = self.dataset_info.edge_types.float()
            e_marginals = edge_types / torch.sum(edge_types)
            print(f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges")
            self.transition_model = MarginalUniformTransition(x_marginals=x_marginals, e_marginals=e_marginals,
                                                              y_classes=self.ydim_output)
            self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals,
                                                y=torch.ones(self.ydim_output) / self.ydim_output)

        self.save_hyperparameters(ignore=[train_metrics, sampling_metrics])
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_loss = 1e8
        self.val_counter = 0

    def training_step(self, data, i):
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        
        # One-hot encode data for diffusion
        ohe_X = []
        for i, dim in enumerate(self.dataset_info.node_feature_dims):
            ohe_X.append(F.one_hot(dense_data.X[..., i], num_classes=dim))
        X_one_hot = torch.cat(ohe_X, dim=-1).float()

        ohe_E = []
        for i, dim in enumerate(self.dataset_info.edge_feature_dims):
            ohe_E.append(F.one_hot(dense_data.E[..., i], num_classes=dim))
        E_one_hot = torch.cat(ohe_E, dim=-1).float()
        
        ohe_y = []
        for i, dim in enumerate(self.dataset_info.y_feature_dims):
            if data.y[..., i].max() >= dim:
                raise ValueError(f"Y feature {i} has value out of bounds: {data.y[..., i].max()} >= {dim}")
            ohe_y.append(F.one_hot(data.y[..., i], num_classes=dim))
        y_one_hot = torch.cat(ohe_y, dim=-1).float()
        noisy_data = self.apply_noise(X_one_hot, E_one_hot, y_one_hot, node_mask)

        if hasattr(data, 'x_continuous'):
            # dense_x_continuous, _ = utils.to_dense_batch(x=data.x_continuous, batch=data.batch)
            # noisy_data['X_continuous'] = dense_x_continuous
            pass
        
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        loss = self.train_loss(masked_pred_X=pred.X, masked_pred_E=pred.E, pred_y=pred.y,
                               true_X=X_one_hot, true_E=E_one_hot, true_y=y_one_hot,
                               log=i % self.log_every_steps == 0)

        self.train_metrics(masked_pred_X=pred.X, masked_pred_E=pred.E, true_X=X_one_hot, true_E=E_one_hot,
                           log=i % self.log_every_steps == 0)

        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True,
                                 weight_decay=self.cfg.train.weight_decay)

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        print("Size of the input features", self.Xdim, self.Edim, self.ydim)

    def on_train_epoch_start(self) -> None:
        print("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        self.train_loss.log_epoch_metrics(self.current_epoch, self.start_epoch_time)
        self.train_metrics.log_epoch_metrics(self.current_epoch)

    def on_validation_epoch_start(self) -> None:
        self.val_nll.reset()
        self.val_CE.reset()
        self.val_X_kl.reset()
        self.val_E_kl.reset()
        self.val_y_kl.reset()
        self.val_X_logp.reset()
        self.val_E_logp.reset()
        self.val_y_logp.reset()

    def validation_step(self, data, i):
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)

        # One-hot encode data for diffusion
        ohe_X = []
        for i, dim in enumerate(self.dataset_info.node_feature_dims):
            ohe_X.append(F.one_hot(dense_data.X[..., i], num_classes=dim))
        X_one_hot = torch.cat(ohe_X, dim=-1).float()

        ohe_E = []
        for i, dim in enumerate(self.dataset_info.edge_feature_dims):
            ohe_E.append(F.one_hot(dense_data.E[..., i], num_classes=dim))
        E_one_hot = torch.cat(ohe_E, dim=-1).float()
        
        ohe_y = []
        for i, dim in enumerate(self.dataset_info.y_feature_dims):
            if data.y[..., i].max() >= dim or data.y[..., i].min() < 0:
                raise ValueError(f"Invalid y field {i}: min={data.y[...,i].min()}, max={data.y[...,i].max()}, dim={dim}")
            ohe_y.append(F.one_hot(data.y[..., i], num_classes=dim))
        y_one_hot = torch.cat(ohe_y, dim=-1).float()

        noisy_data = self.apply_noise(X_one_hot, E_one_hot, y_one_hot, node_mask)

        if hasattr(data, 'x_continuous'):
            # dense_x_continuous, _ = utils.to_dense_batch(x=data.x_continuous, batch=data.batch)
            # noisy_data['X_continuous'] = dense_x_continuous
            pass

        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        # Compute and log stable CE loss
        ce_loss = self.train_loss(masked_pred_X=pred.X, masked_pred_E=pred.E, pred_y=pred.y,
                                  true_X=X_one_hot, true_E=E_one_hot, true_y=y_one_hot,
                                  log=False)
        self.val_CE.update(ce_loss)

        # Compute NLL loss for observation
        self.compute_val_loss(pred, noisy_data, X_one_hot, E_one_hot, y_one_hot, node_mask, test=False)
        
        return {'loss': ce_loss}

    def on_validation_epoch_end(self) -> None:
        # Log CE loss
        val_ce = self.val_CE.compute()
        self.log("val/epoch_CE_loss", val_ce)
        wandb.log({"val/epoch_CE_loss": val_ce}, commit=False)

        # Log NLL loss
        metrics = [self.val_nll.compute(), self.val_X_kl.compute(), self.val_E_kl.compute(),
                   self.val_X_logp.compute(), self.val_E_logp.compute(),
                   self.val_y_logp.compute()]
        wandb.log({"val/epoch_NLL": metrics[0],
                   "val/X_kl": metrics[1],
                   "val/E_kl": metrics[2],
                   "val/X_logp": metrics[3],
                   "val/E_logp": metrics[4],
                   "val/y_logp": metrics[5]}, commit=False)

        print(f"Epoch {self.current_epoch}: Val CE {val_ce :.2f} -- Val NLL {metrics[0] :.2f}\n")

        # Log val_CE with default Lightning logger for checkpointing
        self.log("val_loss", val_ce)

        if val_ce < self.best_val_loss:
            self.best_val_loss = val_ce
        print('Val CE loss: %.4f 	 Best val CE loss:  %.4f\n' % (val_ce, self.best_val_loss))

        self.val_counter += 1
        if self.val_counter % self.cfg.general.sample_every_val == 0:
            start = time.time()
            samples_left_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_save = self.cfg.general.samples_to_save
            chains_left_to_save = self.cfg.general.chains_to_save

            samples = []

            val_batch = next(iter(self.trainer.datamodule.val_dataloader()))
            y = val_batch.y.to(self.device)

            ident = 0
            while samples_left_to_generate > 0:
                bs = min(8, 2 * self.cfg.train.batch_size)
                to_generate = min(samples_left_to_generate, bs)
                to_save = min(samples_left_to_save, bs)
                chains_save = min(chains_left_to_save, bs)
                y_slice = y[:to_generate]
                if y_slice.size(0) < to_generate:
                    y_slice = y_slice.repeat(to_generate // y_slice.size(0) + 1, 1)[:to_generate]

                samples.extend(self.sample_batch(batch_id=ident, batch_size=to_generate, num_nodes=None,
                                                 save_final=to_save,
                                                 keep_chain=chains_save,
                                                 number_chain_steps=self.number_chain_steps, y=y_slice))
                ident += to_generate

                samples_left_to_save -= to_save
                samples_left_to_generate -= to_generate
                chains_left_to_save -= chains_save
            print("Computing sampling metrics...")
            self.sampling_metrics(samples, self.name, self.current_epoch, val_counter=-1, test=False)
            print(f'Done. Sampling took {time.time() - start:.2f} seconds\n')
            self.sampling_metrics.reset()

    def on_test_epoch_start(self) -> None:
        self.test_nll.reset()
        self.test_X_kl.reset()
        self.test_E_kl.reset()
        self.test_X_logp.reset()
        self.test_E_logp.reset()
        self.test_y_logp.reset()

    def test_step(self, data, i):
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)

        # One-hot encode data for diffusion
        ohe_X = []
        for i, dim in enumerate(self.dataset_info.node_feature_dims):
            ohe_X.append(F.one_hot(dense_data.X[..., i], num_classes=dim))
        X_one_hot = torch.cat(ohe_X, dim=-1).float()

        ohe_E = []
        for i, dim in enumerate(self.dataset_info.edge_feature_dims):
            ohe_E.append(F.one_hot(dense_data.E[..., i], num_classes=dim))
        E_one_hot = torch.cat(ohe_E, dim=-1).float()
        
        ohe_y = []
        for i, dim in enumerate(self.dataset_info.y_feature_dims):
            if data.y[..., i].max() >= dim or data.y[..., i].min() < 0:
                raise ValueError(f"Invalid y field {i}: min={data.y[...,i].min()}, max={data.y[...,i].max()}, dim={dim}")
            ohe_y.append(F.one_hot(data.y[..., i], num_classes=dim))
        y_one_hot = torch.cat(ohe_y, dim=-1).float()

        noisy_data = self.apply_noise(X_one_hot, E_one_hot, y_one_hot, node_mask)

        if hasattr(data, 'x_continuous'):
            # dense_x_continuous, _ = utils.to_dense_batch(x=data.x_continuous, batch=data.batch)
            # noisy_data['X_continuous'] = dense_x_continuous
            pass
            
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        nll = self.compute_val_loss(pred, noisy_data, X_one_hot, E_one_hot, y_one_hot, node_mask, test=True)
        return {'loss': nll}

    def on_test_epoch_end(self) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        metrics = [self.test_nll.compute(), self.test_X_kl.compute(), self.test_E_kl.compute(),
                   self.test_X_logp.compute(), self.test_E_logp.compute(),
                   self.test_y_logp.compute()]
        wandb.log({"test/epoch_NLL": metrics[0],
                   "test/X_mse": metrics[1],
                   "test/E_mse": metrics[2],
                   "test/X_logp": metrics[3],
                   "test/E_logp": metrics[4],
                   "test/y_logp": metrics[5]}, commit=False)

        print(f"Epoch {self.current_epoch}: Test NLL {metrics[0] :.2f} -- Test Atom type KL {metrics[1] :.2f} -- "
              f"Test Edge type KL: {metrics[2] :.2f}\n")

        test_nll = metrics[0]
        wandb.log({"test/epoch_NLL": test_nll}, commit=False)

        print(f'Test loss: {test_nll :.4f}')

        samples_left_to_generate = self.cfg.general.final_model_samples_to_generate
        samples_left_to_save = self.cfg.general.final_model_samples_to_save
        chains_left_to_save = self.cfg.general.final_model_chains_to_save

        samples = []
        test_batch = next(iter(self.trainer.datamodule.test_dataloader()))
        y = test_batch.y.to(self.device)

        id = 0
        while samples_left_to_generate > 0:
            print(f'Samples left to generate: {samples_left_to_generate}/'
                  f'{self.cfg.general.final_model_samples_to_generate}', end='', flush=True)
            bs = 2 * self.cfg.train.batch_size
            to_generate = min(samples_left_to_generate, bs)
            to_save = min(samples_left_to_save, bs)
            chains_save = min(chains_left_to_save, bs)
            y_slice = y[:to_generate]
            if y_slice.size(0) < to_generate:
                y_slice = y_slice.repeat(to_generate // y_slice.size(0) + 1, 1)[:to_generate]

            samples.extend(self.sample_batch(id, to_generate, num_nodes=None, save_final=to_save,
                                             keep_chain=chains_save, number_chain_steps=self.number_chain_steps, y=y_slice))
            id += to_generate
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save
        print("Computing sampling metrics...")
        self.sampling_metrics.reset()
        self.sampling_metrics(samples, self.name, self.current_epoch, self.val_counter, test=True)
        self.sampling_metrics.reset()
        print("Done.")


    def kl_prior(self, X, E, y, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1)."""
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((X.size(0), 1), device=X.device)
        Ts = self.T * ones
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)

        # Compute transition probabilities
        probX = X @ Qtb.X
        probE = E @ Qtb.E.unsqueeze(1)

        bs, n, _ = probX.shape

        limit_X = self.limit_dist.X[None, None, :].expand(bs, n, -1).type_as(probX)
        limit_E = self.limit_dist.E[None, None, None, :].expand(bs, n, n, -1).type_as(probE)

        limit_dist_X, limit_dist_E, probX, probE = diffusion_utils.mask_distributions(
            true_X=limit_X.clone(), true_E=limit_E.clone(),
            pred_X=probX, pred_E=probE, node_mask=node_mask
        )

        # FIX: Add epsilon before taking log for numerical stability
        probX = probX + 1e-8
        probX = probX / probX.sum(dim=-1, keepdim=True)

        probE = probE + 1e-8
        probE = probE / probE.sum(dim=-1, keepdim=True)

        kl_distance_X = F.kl_div(input=probX.log(), target=limit_dist_X, reduction='none')
        kl_distance_E = F.kl_div(input=probE.log(), target=limit_dist_E, reduction='none')

        kl_distance_y = 0
        return diffusion_utils.sum_except_batch(kl_distance_X) + \
            diffusion_utils.sum_except_batch(kl_distance_E) + \
            kl_distance_y

    def compute_Lt(self, X, E, y, pred, noisy_data, node_mask, test):
        pred_probs_X = F.softmax(pred.X, dim=-1)
        pred_probs_E = F.softmax(pred.E, dim=-1)
        pred_probs_y = F.softmax(pred.y, dim=-1)

        Qtb = self.transition_model.get_Qt_bar(noisy_data['alpha_t_bar'], self.device)
        Qsb = self.transition_model.get_Qt_bar(noisy_data['alpha_s_bar'], self.device)
        Qt = self.transition_model.get_Qt(noisy_data['beta_t'], self.device)

        # Compute distributions to compare with KL
        bs, n, d = X.shape
        prob_true = diffusion_utils.posterior_distributions(X=X, E=E, y=y, X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_true.E = prob_true.E.reshape((bs, n, n, -1))
        prob_pred = diffusion_utils.posterior_distributions(X=pred_probs_X, E=pred_probs_E, y=pred_probs_y,
                                                            X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_pred.E = prob_pred.E.reshape((bs, n, n, -1))

        # Reshape and filter masked rows
        prob_true_X, prob_true_E, prob_pred.X, prob_pred.E = diffusion_utils.mask_distributions(true_X=prob_true.X,
                                                                                                true_E=prob_true.E,
                                                                                                pred_X=prob_pred.X,
                                                                                                pred_E=prob_pred.E,
                                                                                                node_mask=node_mask)
        
        # FIX: Add epsilon before taking log for numerical stability
        prob_pred.X = prob_pred.X + 1e-7
        prob_pred.X = prob_pred.X / prob_pred.X.sum(dim=-1, keepdim=True)

        prob_pred.E = prob_pred.E + 1e-7
        prob_pred.E = prob_pred.E / prob_pred.E.sum(dim=-1, keepdim=True)

        # FIX: Add epsilon to target distributions and re-normalize for numerical stability
        prob_true.X = prob_true.X + 1e-7
        prob_true.X = prob_true.X / prob_true.X.sum(dim=-1, keepdim=True)

        prob_true.E = prob_true.E + 1e-7
        prob_true.E = prob_true.E / prob_true.E.sum(dim=-1, keepdim=True)

        kl_x = (self.test_X_kl if test else self.val_X_kl)(torch.log(prob_pred.X), prob_true.X)
        kl_e = (self.test_E_kl if test else self.val_E_kl)(torch.log(prob_pred.E), prob_true.E)
        
        return kl_x + kl_e
    def reconstruction_logp(self, t, X, E, y, node_mask):
        # Compute noise values for t = 0.
        t_zeros = torch.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=self.device)

        probX0 = X @ Q0.X  # (bs, n, dx_out)
        probE0 = E @ Q0.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled0 = diffusion_utils.sample_discrete_features(probX=probX0, probE=probE0, node_mask=node_mask)

        X0 = F.one_hot(sampled0.X, num_classes=self.Xdim_output).float()
        E0 = F.one_hot(sampled0.E, num_classes=self.Edim_output).float()
        y0 = y
        assert (X.shape == X0.shape) and (E.shape == E0.shape)

        sampled_0 = utils.PlaceHolder(X=X0, E=E0, y=y0).mask(node_mask)

        # Predictions
        noisy_data = {'X_t': sampled_0.X, 'E_t': sampled_0.E, 'y_t': sampled_0.y, 'node_mask': node_mask,
                      't': torch.zeros(X0.shape[0], 1).type_as(y0)}
        extra_data = self.compute_extra_data(noisy_data)
        pred0 = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        probX0 = F.softmax(pred0.X, dim=-1)
        probE0 = F.softmax(pred0.E, dim=-1)
        proby0 = F.softmax(pred0.y, dim=-1)

        # Set masked rows to arbitrary values that don't contribute to loss
        probX0[~node_mask] = torch.ones(self.Xdim_output).type_as(probX0)
        probE0[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))] = torch.ones(self.Edim_output).type_as(probE0)

        diag_mask = torch.eye(probE0.size(1)).type_as(probE0).bool()
        diag_mask = diag_mask.unsqueeze(0).expand(probE0.size(0), -1, -1)
        probE0[diag_mask] = torch.ones(self.Edim_output).type_as(probE0)

        # Add epsilon to avoid log(0)
        probX0 = probX0 + 1e-7
        probE0 = probE0 + 1e-7
        proby0 = proby0 + 1e-7
        # Re-normalize after adding epsilon                                                                                                                
        probX0 = probX0 / probX0.sum(dim=-1, keepdim=True)                                                                                                 
        probE0 = probE0 / probE0.sum(dim=-1, keepdim=True)                                                                                                 
        proby0 = proby0 / proby0.sum(dim=-1, keepdim=True)                                                                                                 
                                                                                                                                                        
        return utils.PlaceHolder(X=probX0, E=probE0, y=proby0)

    def apply_noise(self, X, E, y, node_mask):
        """ Sample noise and apply it to the data. """

        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = torch.einsum('bijk, bkl->bijl', E, Qtb.E)  # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output).float()
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output).float()
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': y, 'node_mask': node_mask}
        return noisy_data

    def compute_val_loss(self, pred, noisy_data, X, E, y, node_mask, test=False):
        """Computes an estimator for the variational lower bound, or the simple loss (MSE).
           pred: (batch_size, n, total_features)
           noisy_data: dict
           X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
           node_mask : (bs, n)
           Output: nll (size 1)
       """
        t = noisy_data['t']

        # 1.
        N = node_mask.sum(1).long()
        log_pN = self.node_dist.log_prob(N)

        # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
        kl_prior = self.kl_prior(X, E, y, node_mask)

        # 3. Diffusion loss
        loss_all_t = self.compute_Lt(X, E, y, pred, noisy_data, node_mask, test)

        # 4. Reconstruction loss
        # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
        prob0 = self.reconstruction_logp(t, X, E, y, node_mask)

        loss_term_0 = self.val_X_logp(X * prob0.X.log()) + self.val_E_logp(E * prob0.E.log()) + self.val_y_logp(y * prob0.y.log())
                                                                                                                                                             
        # Combine terms                                                                                                                                    
        nlls = - log_pN + kl_prior + loss_all_t - loss_term_0                                                                                              
        assert len(nlls.shape) == 1, f'{nlls.shape} has more than only batch dim.'   
        # Update NLL metric object and return batch nll
        nll = (self.test_nll if test else self.val_nll)(nlls)        # Average over the batch

        wandb.log({"kl prior": kl_prior.mean(),
                   "Estimator loss terms": loss_all_t.mean(),
                   "log_pn": log_pN.mean(),
                   "loss_term_0": loss_term_0,
                   'test_nll' if test else 'val_nll': nll}, commit=False)

    def forward(self, noisy_data, extra_data, node_mask):
        # Convert one-hot noisy data to integer indices
        x_noisy_one_hot = noisy_data['X_t']
        x_dims = self.dataset_info.node_feature_dims
        x_int_fields = []
        current_dim = 0
        for dim in x_dims:
            feature_one_hot = x_noisy_one_hot[..., current_dim:current_dim+dim]
            feature_int = torch.argmax(feature_one_hot, dim=-1)
            x_int_fields.append(feature_int.unsqueeze(-1))
            current_dim += dim
        x_int = torch.cat(x_int_fields, dim=-1)

        e_noisy_one_hot = noisy_data['E_t']
        e_dims = self.dataset_info.edge_feature_dims
        e_int_fields = []
        current_dim = 0
        for dim in e_dims:
            feature_one_hot = e_noisy_one_hot[..., current_dim:current_dim+dim]
            feature_int = torch.argmax(feature_one_hot, dim=-1)
            e_int_fields.append(feature_int.unsqueeze(-1))
            current_dim += dim
        e_int = torch.cat(e_int_fields, dim=-1)

        y_noisy_one_hot = noisy_data['y_t']
        y_dims = self.dataset_info.y_feature_dims
        if len(y_dims) == 0:
            raise ValueError("No y features found — preprocessing bug.")
        y_int_fields = []
        current_dim = 0
        for dim in y_dims:
            feature_one_hot = y_noisy_one_hot[..., current_dim:current_dim+dim]
            feature_int = torch.argmax(feature_one_hot, dim=-1)
            y_int_fields.append(feature_int.unsqueeze(-1))
            current_dim += dim
        y_int = torch.cat(y_int_fields, dim=-1)

        # Embed the integer indices
        x_embed = self.model.embed_X(x_int)
        e_embed = self.model.embed_E(e_int)
        y_embed = self.model.embed_y(y_int)

        X = x_embed.float()
        E = e_embed.float()
        y = y_embed.float()
        return self.model(X, E, y, node_mask)

    @torch.no_grad()
    def sample_batch(self, batch_id: int, batch_size: int, keep_chain: int, number_chain_steps: int,
                     save_final: int, num_nodes=None, y=None):
            """
            :param batch_id: int
            :param batch_size: int
            :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
            :param save_final: int: number of predictions to save to file
            :param keep_chain: int: number of chains to save to file
            :param number_chain_steps: number of timesteps to save for each chain
            :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
            """
            if num_nodes is None:
                n_nodes = self.node_dist.sample_n(batch_size, self.device)
            elif type(num_nodes) == int:
                n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
            else:
                assert isinstance(num_nodes, torch.Tensor)
                n_nodes = num_nodes
            n_max = torch.max(n_nodes).item()

            # Build the masks
            arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
            node_mask = arange < n_nodes.unsqueeze(1)
            # TODO: how to move node_mask on the right device in the multi-gpu case?
            # TODO: everything else depends on its device
            # Sample noise  -- z has size (n_samples, n_nodes, n_features)
            z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
            X, E = z_T.X, z_T.E
    
            if y is None:
                y = z_T.y
            else:
                # y is integer encoded, so one-hot encode it
                ohe_y = []
                for i, dim in enumerate(self.dataset_info.y_feature_dims):
                    if y[..., i].max() >= dim or y[..., i].min() < 0:
                        raise ValueError(f"Invalid y field {i}: min={y[...,i].min()}, max={y[...,i].max()}, dim={dim}")
                    ohe_y.append(F.one_hot(y[..., i], num_classes=dim))
                y = torch.cat(ohe_y, dim=-1).float()
    
            assert (E == torch.transpose(E, 1, 2)).all()
            assert number_chain_steps < self.T
    
            # Initialize chains only if they are required
            if keep_chain > 0 and number_chain_steps > 0:
                chain_X_size = torch.Size((number_chain_steps, keep_chain, X.size(1)))
                chain_E_size = torch.Size((number_chain_steps, keep_chain, E.size(1), E.size(2)))
                chain_X = torch.zeros(chain_X_size)
                chain_E = torch.zeros(chain_E_size)
            else:
                chain_X = None
                chain_E = None
    
            # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
            for s_int in reversed(range(0, self.T)):
                s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
                t_array = s_array + 1
                s_norm = s_array / self.T
                t_norm = t_array / self.T
    
                # Sample z_s
                sampled_s, discrete_sampled_s, predicted_graph = self.sample_p_zs_given_zt(t_norm, X, E, y, node_mask,
                                                                                           last_step=s_int == 100)
                X, E, y = sampled_s.X, sampled_s.E, sampled_s.y
    
                # Save the first keep_chain graphs
                if chain_X is not None:
                    write_index = (s_int * number_chain_steps) // self.T
                    chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
                    chain_E[write_index] = discrete_sampled_s.E[:keep_chain]
    
            # Sample
            sampled_s = sampled_s.mask(node_mask, collapse=True)
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y
    
    
    
            # Prepare the chain for saving
            if chain_X is not None and keep_chain > 0:
                final_X_chain = X[:keep_chain]
                final_E_chain = E[:keep_chain]
    
                chain_X[0] = final_X_chain                  # Overwrite last frame with the resulting X, E
                chain_E[0] = final_E_chain
    
                chain_X = diffusion_utils.reverse_tensor(chain_X)
                chain_E = diffusion_utils.reverse_tensor(chain_E)
    
                # Repeat last frame to see final sample better
                chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
                chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
                assert chain_X.size(0) == (number_chain_steps + 10)
    
            # Clamp is used to be safe, in case of -1 values for masked nodes.
            X_one_hot = F.one_hot(torch.clamp(X.long(), min=0), num_classes=self.Xdim_output).float()
            E_one_hot = F.one_hot(torch.clamp(E.long(), min=0), num_classes=self.Edim_output).float()
    
            molecule_list = []
            for i in range(batch_size):
                n = n_nodes[i]
                atom_types = X_one_hot[i, :n].cpu()
                edge_types = E_one_hot[i, :n, :n].cpu()
                molecule_list.append([atom_types, edge_types])
    
            predicted_graph_list = []
            for i in range(batch_size):
                n = n_nodes[i]
                atom_types = X_one_hot[i, :n].cpu()
                edge_types = E_one_hot[i, :n, :n].cpu()
                predicted_graph_list.append([atom_types, edge_types])
    
    
            # Save generated graphs for analysis
            if save_final > 0:
                save_path = f'graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}_generated_graphs.pt'
                torch.save(molecule_list, save_path)
    
            # Visualize chains
            if self.visualization_tools is not None and chain_X is not None:
                print('Visualizing chains...')
                # One-hot encode for visualization
                chain_X_one_hot = F.one_hot(torch.clamp(chain_X.long(), min=0), num_classes=self.Xdim_output).float()
                chain_E_one_hot = F.one_hot(torch.clamp(chain_E.long(), min=0), num_classes=self.Edim_output).float()
    
                current_path = os.getcwd()
                num_molecules = chain_X.size(1)       # number of molecules
                for i in range(num_molecules):
                    result_path = os.path.join(current_path, f'chains/{self.cfg.general.name}/'
                                                             f'epoch{self.current_epoch}/'
                                                             f'chains/molecule_{batch_id + i}')
                    if not os.path.exists(result_path):
                        os.makedirs(result_path)
                        _ = self.visualization_tools.visualize_chain(result_path,
                                                                     chain_X_one_hot[:, i, :, :],
                                                                     chain_E_one_hot[:, i, :, :, :])
                    print('\r{}/{} complete'.format(i+1, num_molecules), end='', flush=True)
                print('\nVisualizing molecules...')
    
                # Visualize the final molecules
                current_path = os.getcwd()
                result_path = os.path.join(current_path,
                                           f'graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/')
                self.visualization_tools.visualize(result_path, molecule_list, save_final)
                self.visualization_tools.visualize(result_path, predicted_graph_list, save_final, log='predicted')
                print("Done.")
    
            return molecule_list
    def sample_p_zs_given_zt(self, t, X_t, E_t, y_t, node_mask, last_step: bool):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        bs, n, dxs = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}
        
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)               # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)               # bs, n, n, d0

        if last_step:
            predicted_graph = diffusion_utils.sample_discrete_features(pred_X, pred_E, node_mask=node_mask)

        p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=X_t,
                                                                                           Qt=Qt.X,
                                                                                           Qsb=Qsb.X,
                                                                                           Qtb=Qtb.X)

        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=E_t,
                                                                                           Qt=Qt.E,
                                                                                           Qsb=Qsb.E,
                                                                                           Qtb=Qtb.E)
        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X         # bs, n, d0, d_t-1
        unnormalized_prob_X = weighted_X.sum(dim=2)                     # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E        # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask=node_mask)

        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=y_t)
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=y_t)

        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t), \
               predicted_graph if last_step else None

    def compute_extra_data(self, noisy_data):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """

        extra_features = self.extra_features(noisy_data)
        return extra_features
