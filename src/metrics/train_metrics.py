import torch
from torch import Tensor
import torch.nn as nn
from torchmetrics import Metric, MeanSquaredError, MetricCollection
import time
import wandb
from src.metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchMSE, SumExceptBatchKL, CrossEntropyMetric, \
    ProbabilityMetric, NLL


class NodeMSE(MeanSquaredError):
    def __init__(self, *args):
        super().__init__(*args)


class EdgeMSE(MeanSquaredError):
    def __init__(self, *args):
        super().__init__(*args)


class TrainLoss(nn.Module):
    def __init__(self):
        super(TrainLoss, self).__init__()
        self.train_node_mse = NodeMSE()
        self.train_edge_mse = EdgeMSE()
        self.train_y_mse = MeanSquaredError()

    def forward(self, masked_pred_epsX, masked_pred_epsE, pred_y, true_epsX, true_epsE, true_y, log: bool):
        mse_X = self.train_node_mse(masked_pred_epsX, true_epsX) if true_epsX.numel() > 0 else 0.0
        mse_E = self.train_edge_mse(masked_pred_epsE, true_epsE) if true_epsE.numel() > 0 else 0.0
        mse_y = self.train_y_mse(pred_y, true_y) if true_y.numel() > 0 else 0.0
        mse = mse_X + mse_E + mse_y

        if log:
            to_log = {'train_loss/batch_mse': mse.detach(),
                      'train_loss/node_MSE': self.train_node_mse.compute(),
                      'train_loss/edge_MSE': self.train_edge_mse.compute(),
                      'train_loss/y_mse': self.train_y_mse.compute()}
            wandb.log(to_log, commit=True)

        return mse

    def reset(self):
        for metric in (self.train_node_mse, self.train_edge_mse, self.train_y_mse):
            metric.reset()

    def log_epoch_metrics(self, current_epoch, start_epoch_time):
        epoch_node_mse = self.train_node_mse.compute() if self.train_node_mse.total > 0 else -1
        epoch_edge_mse = self.train_edge_mse.compute() if self.train_edge_mse.total > 0 else -1
        epoch_y_mse = self.train_y_mse.compute() if self.train_y_mse.total > 0 else -1

        to_log = {"train_epoch/epoch_X_mse": epoch_node_mse,
                  "train_epoch/epoch_E_mse": epoch_edge_mse,
                  "train_epoch/epoch_y_mse": epoch_y_mse}
        print(f"Epoch {current_epoch}: X_mse: {epoch_node_mse :.3f} -- E mse: {epoch_edge_mse :.3f} --"
              f" y_mse: {epoch_y_mse :.3f} -- {time.time() - start_epoch_time:.1f}s ")

        wandb.log(to_log)

        for metric in [self.train_node_mse, self.train_edge_mse, self.train_y_mse]:
            metric.reset()


class TrainLossDiscrete(nn.Module):
    """ Train with Cross entropy"""
    def __init__(self, lambda_train, dataset_infos):
        super().__init__()
        self.lambda_train = lambda_train

        # Create a loss function for each node feature
        self.node_feature_dims = list(dataset_infos.num_node_features.values())
        self.node_losses = nn.ModuleList([CrossEntropyMetric() for _ in self.node_feature_dims])

        # Create a loss function for each edge feature, with class weights
        self.edge_feature_dims = list(dataset_infos.num_edge_features.values())

        # Define weights for the binary flags, based on data analysis
        sequential_weights = torch.tensor([1.0, 10.0])
        close_geo_weights = torch.tensor([1.0, 25.0])
        same_merchant_weights = torch.tensor([1.0, 50.0])

        self.edge_losses = nn.ModuleList()
        for i, feature_name in enumerate(dataset_infos.num_edge_features.keys()):
            if feature_name == 'is_sequential':
                self.edge_losses.append(CrossEntropyMetric(weight=sequential_weights))
            elif feature_name == 'same_merchant_cluster':
                self.edge_losses.append(CrossEntropyMetric(weight=same_merchant_weights))
            elif feature_name == 'is_close_geo':
                self.edge_losses.append(CrossEntropyMetric(weight=close_geo_weights))
            else:
                # For 'time_delta_bin'
                self.edge_losses.append(CrossEntropyMetric())

        # Create a loss function for each y feature
        self.y_feature_dims = list(dataset_infos.num_y_features.values())
        self.y_losses = nn.ModuleList([CrossEntropyMetric() for _ in self.y_feature_dims])

    def forward(self, masked_pred_X, masked_pred_E, true_X, true_E, pred_y, true_y, log: bool):
        # Reshape and mask
        true_X = torch.reshape(true_X, (-1, true_X.size(-1)))
        masked_pred_X = torch.reshape(masked_pred_X, (-1, masked_pred_X.size(-1)))
        mask_X = (true_X != 0.).any(dim=-1)
        flat_true_X = true_X[mask_X, :]
        flat_pred_X = masked_pred_X[mask_X, :]

        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))
        masked_pred_E = torch.reshape(masked_pred_E, (-1, masked_pred_E.size(-1)))
        mask_E = (true_E != 0.).any(dim=-1)
        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]

        # Calculate node loss feature by feature
        total_loss_X = 0
        current_dim = 0
        for i, feature_dim in enumerate(self.node_feature_dims):
            # Slice the tensors for the current feature
            pred_slice = flat_pred_X[:, current_dim : current_dim + feature_dim]
            true_slice = flat_true_X[:, current_dim : current_dim + feature_dim]
            
            # Skip if the slice is empty
            if true_slice.numel() == 0:
                continue

            # Convert true slice to class indices
            true_indices = torch.argmax(true_slice, dim=-1)
            
            # Calculate loss for this feature
            loss = self.node_losses[i](pred_slice, true_indices)
            total_loss_X += loss
            current_dim += feature_dim

        # Calculate edge loss feature by feature
        total_loss_E = 0
        current_dim = 0
        for i, feature_dim in enumerate(self.edge_feature_dims):
            pred_slice = flat_pred_E[:, current_dim : current_dim + feature_dim]
            true_slice = flat_true_E[:, current_dim : current_dim + feature_dim]

            if true_slice.numel() == 0:
                continue

            true_indices = torch.argmax(true_slice, dim=-1)
            loss = self.edge_losses[i](pred_slice, true_indices)
            total_loss_E += loss
            current_dim += feature_dim

        # Calculate y loss feature by feature
        total_loss_y = 0
        current_dim_y = 0
        for i, feature_dim_y in enumerate(self.y_feature_dims):
            pred_slice_y = pred_y[:, current_dim_y : current_dim_y + feature_dim_y]
            true_slice_y = true_y[:, current_dim_y : current_dim_y + feature_dim_y]

            if true_slice_y.numel() == 0:
                continue

            true_indices_y = torch.argmax(true_slice_y, dim=-1)
            loss_y = self.y_losses[i](pred_slice_y, true_indices_y)
            total_loss_y += loss_y
            current_dim_y += feature_dim_y

        if log:
            # Logging logic can be simplified or adjusted as needed
            pass

        return total_loss_X + self.lambda_train[0] * total_loss_E + self.lambda_train[1] * total_loss_y

    def reset(self):
        for loss in self.node_losses:
            loss.reset()
        for loss in self.edge_losses:
            loss.reset()
        for loss in self.y_losses:
            loss.reset()

    def log_epoch_metrics(self, current_epoch, start_epoch_time):
        epoch_node_loss = sum(loss.compute() for loss in self.node_losses if loss.total_samples > 0)
        epoch_edge_loss = sum(loss.compute() for loss in self.edge_losses if loss.total_samples > 0)
        epoch_y_loss = sum(loss.compute() for loss in self.y_losses if loss.total_samples > 0)

        to_log = {"train_epoch/x_CE": epoch_node_loss,
                  "train_epoch/E_CE": epoch_edge_loss,
                  "train_epoch/y_CE": epoch_y_loss}
        wandb.log(to_log, commit=False)

        print(f"Epoch {current_epoch} finished: X: {epoch_node_loss :.2f} -- E: {epoch_edge_loss :.2f} -- Y: {epoch_y_loss :.2f} "
              f"-- {time.time() - start_epoch_time:.1f}s ")



