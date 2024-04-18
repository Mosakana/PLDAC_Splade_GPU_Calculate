import torch
from torch.nn import ReLU
from splade_model import SpladeModel
from copy import deepcopy

batch = 32
length = 10
data_size = 32

vocabulary = 20

list_lengths = torch.randint(0, length, [batch])

print(list_lengths)

x = torch.randn(batch, length, data_size, requires_grad=True, dtype=torch.float64)
w = torch.randn(data_size, vocabulary, requires_grad=True, dtype=torch.float64)
b = torch.randn(vocabulary, requires_grad=True, dtype=torch.float64)

mask = torch.ones(batch, length)


for i, l in enumerate(list_lengths):
    mask[i, l:] = 0


splade = SpladeModel.apply(x, w, b, mask)

splade[0].sum().backward()

grad_splade_x = deepcopy(x.grad)
grad_splade_w = deepcopy(w.grad)
grad_splade_b = deepcopy(b.grad)



###################################################

x.grad = None
w.grad = None
b.grad = None

mask = torch.where(mask == 1, 0, -torch.inf)

relu = ReLU()

out = x @ w
out += b.reshape(1, 1, *b.shape)
out += mask.reshape(*mask.shape, 1)
out = relu(torch.max(out, 1)[0])


out.sum().backward()

#
# print(grad_splade_x)
# print(x.grad)
#
# print(grad_splade_w)
# print(w.grad)
#
# print(grad_splade_b)
# print(b.grad)

print(torch.all(grad_splade_x.isclose(x.grad)))
print(torch.all(grad_splade_w.isclose(w.grad)))
print(torch.all(grad_splade_b.isclose(b.grad)))