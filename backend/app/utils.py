import torch

# CUDA: Compute Unified Device Architecture
# mps: Metal Performance Shaders(Apple製 GPU)
# cpu: 通常のCPU
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)