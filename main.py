import torch
from BigMM.mm import BIGmm
import time

a = torch.ones([1000, 1000]).float()

b = BIGmm(a, a, [97, 97])

c = torch.mm(a, a)
print(c)

d = torch.sum(torch.abs(c - b) > 0.0001)

print(d)

