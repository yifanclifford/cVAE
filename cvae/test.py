import torch

A = torch.rand(5, 4)
B = torch.tensor([[1, 2], [1, 3], [2, 3], [0, 1], [0, 2]])
_, C = torch.topk(A, 2)
print(A)
print(torch.gather(A, 1, C))
print(C)
