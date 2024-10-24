import sys
sys.path.append("../")
import torch
from src.model.cmpt.scot import ScOT, ScOTConfig

model = ScOT.from_pretrained("camlab-ethz/Poseidon-T")
print(model)

ndata1 = torch.randn((64, 4, 64 ,64))
ndata2 = torch.randn((64, 48, 64, 64))
time = torch.randn((64, 1))
print(ndata1.shape)

output = model(pixel_values = ndata1, time = time)
print(output.shape)