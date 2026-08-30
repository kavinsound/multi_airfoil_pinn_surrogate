import torch.nn.functional as F
from neuralop.models import GINO

# Define a mollification weight function
model = GINO(
    in_channels=2,  
    out_channels=9,
    hidden_channels=32,
    in_gno_radius=0.1,
    gno_coord_dim=2,
    gno_weighting_function='bump',
    fno_n_modes=(32, 32),
    fno_n_layers=4,
)