import torch

input_tensor = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).float().reshape(2, 5)
print(input_tensor)

# index_tensor = torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]])
index_tensor = torch.tensor([[0, 1, 0, 1, 0]])
print(index_tensor)

dim = 0
print(torch.gather(input_tensor, dim, index_tensor))
