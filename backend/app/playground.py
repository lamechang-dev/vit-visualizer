import torch
from typing import TypedDict

a = torch.tensor([1, 2])
b = torch.tensor([3, 4])

x = torch.stack([a, b])

print(a)
print(a.shape)

print(x)
print(x.shape)


class Frame(TypedDict):
    index: int
    timestamp_sec: float
    image: str

frame: Frame = {
    "index": 0,
    "timestamp_sec": 1.2,
    "image": "abc"
}