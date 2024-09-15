import torch
import torch.nn as nn

class Identity(nn.Module):
    """Identity encoder that returns the input features as latent features"""
    def forward(self, x):
        return x




        