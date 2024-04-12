import torch
from torch.nn import ReLU
from splade_model import SpladeModel

batch = 64
length = 10
data_size = 32

vocabulary = 20

list_lengths = torch.randint(1, 10, [batch])

print(list_lengths)

x = torch.randn(batch, length, data_size, requires_grad=True, dtype=torch.float64)
w = torch.randn(data_size, vocabulary, requires_grad=True, dtype=torch.float64)
b = torch.randn(vocabulary, requires_grad=True, dtype=torch.float64)

mask = torch.zeros(batch, length)


for i, l in enumerate(list_lengths):
    mask[i, l:] = -torch.inf


splade = SpladeModel.apply(x, w, b, mask)

splade[0].sum().backward()

grad_splade_x = x.grad
grad_splade_w = w.grad
grad_splade_b = b.grad


relu = ReLU()

out = x @ w
out += b.reshape(1, 1, *b.shape)
out += mask.reshape(*mask.shape, 1)
out = relu(torch.max(out, 1)[0])


out.sum().backward()


print(torch.all(grad_splade_x.isclose(x.grad)))
print(torch.all(grad_splade_w.isclose(w.grad)))
print(torch.all(grad_splade_b.isclose(b.grad)))