import torch

a = torch.arange(16).reshape(2, 2, 4)
b = torch.arange(32).reshape(2, 4, 4)

a.unsqueeze_(1)
b.unsqueeze_(2)

c = torch.minimum(a, b)
print(a)
print(b)
print(c)
