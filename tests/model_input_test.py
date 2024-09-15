import sys
sys.path.append('..')

import torch
from architectures import GINO


model = GINO(
    in_channels=1,
    out_channels=1,
    gno_coord_dim=2,
    in_gno_channel_mlp_hidden_layers=[64, 64, 64],
    out_gno_channel_mlp_hidden_layers=[64, 64],
    fno_in_channels=64,
    fno_n_modes=(16, 16),
)

print(model)