import torch

a = torch.tensor([1, 2])
b = torch.tensor([3, 4])

x = torch.stack([a, b])

print(a)
print(a.shape)

print(x)
print(x.shape)