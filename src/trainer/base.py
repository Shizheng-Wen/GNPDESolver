import os
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt

from .optimizers import AdamOptimizer, AdamWOptimizer
from .utils import manual_seed, load_ckpt, save_ckpt, compute_batch_errors, compute_final_metric
from ..data.dataset import Metadata, DATASET_METADATA

class TrainerBase:
    """
    Base class for all trainers, define the init_dataset, initi_model, 
    init_optimizer, train_step, validate, test for coreresponding trainers.

    Attributes:
    ----------
    """
    def __init__(self, args):
        # Config setup
        self.config = args
        self.setup_config = self.config.setup
        self.graph_config = self.config.graph
        self.model_config = self.config.model
        self.dataset_config = self.config.dataset
        self.optimizer_config = self.config.optimizer
        self.path_config = self.config.path

        self.metadata = DATASET_METADATA[self.dataset_config["metaname"]]

        self.device = self.setup_config.device
        manual_seed(self.setup_config.seed)
        if self.setup_config.dtype == "float" or \
            self.setup_config.dtype == "torch.float32" or \
            self.setup_config.dtype == "torch.FloatTensor":
            self.dtype = torch.float32
        elif self.setup_config.dtype == "dobule" or \
            self.setup_config.dtype == "torch.float64" or \
            self.setup_config.dtype == "torch.DoubleTensor":
            self.dtype = torch.float64
        else:
            raise ValueError(f"Invalid dtype: {self.setup_config.dtype}")
        self.loss_fn = nn.MSELoss()
        
        self.init_dataset(self.dataset_config)
        self.init_graph(self.graph_config)
        self.init_model(self.model_config)
        self.init_optimizer(self.optimizer_config)

        nparam = sum(
            [p.numel() * 2 if p.is_complex() else p.numel() for p in self.model.parameters()]
        )
        nbytes = sum(
            [p.numel() * 2 * p.element_size() if p.is_complex() else p.numel() * p.element_size() for p in self.model.parameters()]
        )
        print(f"Number of parameters: {nparam}")
        args.datarow['nparams'] = nparam
        args.datarow['nbytes'] = nbytes

# ------------ init ------------ #
    def init_dataset(self, dataset_config):
        raise NotImplementedError
    
    def init_model(self, model_config):
        raise NotImplementedError
    
    def init_graph(self, graph_config):
        raise NotImplementedError

    def init_optimizer(self, optimizer_config):
        """Initialize the optimizer"""

        self.optimizer = {
            "adam": AdamOptimizer,
            "adamw":AdamWOptimizer
        }[self.optimizer_config["name"]](self.model.parameters(), self.optimizer_config["args"])

# ------------ utils ------------ #
    def to(self, device):
        self.model.to(device)
    
    def type(self, dtype):
        # TODO: check if this is necessary, dataloader does not have type method
        self.model.type(dtype)
        self.train_loader.type(dtype)
        self.val_loader.type(dtype)
        self.test_loader.type(dtype)

    def load_ckpt(self):
        load_ckpt(self.path_config["ckpt_path"], model = self.model)
        return self
    
    def save_ckpt(self):
        """Save checkpoint to the config.ckpt_path"""
        os.makedirs(os.path.dirname(self.path_config["ckpt_path"]), exist_ok=True)
        save_ckpt(self.path_config["ckpt_path"], model = self.model)

        return self

    def compute_test_errors(self):
        # TODO: compute test errors (need to modulate based on the type of dataset, based on metadata)
        raise NotImplementedError

# ------------ train ------------ #
    def train_step(self, batch):
        x_batch, y_batch = batch
        x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
        x_batch, y_batch = x_batch.squeeze(1), y_batch.squeeze(1)
        pred = self.model(self.rigraph, x_batch)
        return self.loss_fn(pred, y_batch)
    
    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                x_batch, y_batch = x_batch.squeeze(1), y_batch.squeeze(1)
                pred = self.model(self.rigraph, x_batch)
                loss = self.loss_fn(pred, y_batch)
                total_loss += loss.item()
        return total_loss / len(loader)

    def fit(self, verbose=False):
        self.to(self.device)
        #self.type(self.dtype)

        result = self.optimizer.optimize(self)
        self.config.datarow['training time'] = result['time']
        
        self.save_ckpt()

        if len(result['train']['loss'])==0:
            if self.config.use_variance_test:
                self.variance_test()
            else:
                self.test()
        else:
            kwargs = {
                "epochs":result['train']['epoch'],
                "losses":result['train']['loss']
            }
        
            if "valid" in result:
                kwargs['val_epochs'] = result['valid']['epoch']
                kwargs['val_losses']= result['valid']['loss']
            
            if "best" in result:
                kwargs['best_epoch'] = result['best']['epoch']
                kwargs['best_loss']  = result['best']['loss']
            
            self.plot_losses(
                **kwargs
            )

            if self.setup_config["use_variance_test"]:
                self.variance_test()
            else:
                self.test()

# ------------ plot ------------ #
    def plot_losses(self, 
                    epochs,
                    losses, 
                    val_epochs = None,
                    val_losses = None,
                    best_epoch = None,
                    best_loss  = None):
        
        if val_losses is None:
            # plot only train loss
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(epochs, losses)
            ax.scatter([best_epoch],[best_loss], c='r', marker='o', label="best loss")
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Loss vs Epoch')
            ax.legend()
            ax.set_xlim(left=0)
            if (np.array(losses) > 0).all():
                ax.set_yscale('log')
            np.savez(self.path_config["loss_path"][:-4]+".npz", epochs=epochs, losses=losses)

        else:
            # also plot valid loss
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            
            ax[0].plot(epochs, losses)
            ax[0].scatter([best_epoch],[best_loss], c='r', marker='o', label="best loss")
            ax[0].set_xlabel('Epoch')
            ax[0].set_ylabel('Loss')
            ax[0].set_title('Loss vs Epoch')
            ax[0].legend()
            ax[0].set_xlim(left=0)
            if (np.array(losses) > 0).all():
                ax[0].set_yscale('log')

            ax[1].plot(val_epochs, val_losses)
            # if best_epoch is not None and best_loss is not None:
            #     ax[1].scatter([best_epoch],[best_loss], c='r', marker='o', label="best validation loss")
            ax[1].set_xlabel('Epoch')
            ax[1].set_ylabel('relative error')
            ax[1].set_title('Loss vs relative error')
            ax[1].legend()
            ax[1].set_xlim(left=0)
            if (np.array(val_losses) > 0).all():
                ax[1].set_yscale('log')
            plt.savefig(self.path_config["loss_path"])
            np.savez(self.path_config["loss_path"][:-4]+".npz", epochs=epochs, losses=losses, val_epochs=val_epochs, val_losses=val_losses)

    def plot_results(self):
        raise NotImplementedError

# ------------ test ------------ #
    def variance_test(self):
        raise NotImplementedError
    
    def test(self):
        raise NotImplementedError
    
        
