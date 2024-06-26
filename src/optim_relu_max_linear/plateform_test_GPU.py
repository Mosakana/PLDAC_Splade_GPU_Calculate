import torch
from optim_relu_max_linear import OptimReluMaxLinear
from copy import deepcopy
from torch.nn import ReLU


batch = 16
length = 100
data_size = 300

vocabulary = 1000

list_lengths = torch.randint(0, length, [batch])


x = torch.randn(batch, length, data_size, requires_grad=True, dtype=torch.float64, device='cuda')
w = torch.randn(data_size, vocabulary, requires_grad=True, dtype=torch.float64, device='cuda')
b = torch.randn(vocabulary, requires_grad=True, dtype=torch.float64, device='cuda')

mask = torch.ones(batch, length).cuda()
for i, l in enumerate(list_lengths):
    mask[i, l:] = 0

splade = OptimReluMaxLinear.apply(x, w, b, mask)

splade.sum().backward()

grad_splade_w = deepcopy(w.grad)
grad_splade_x = deepcopy(x.grad)
grad_splade_b = deepcopy(b.grad)

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

print(torch.all(grad_splade_w.isclose(w.grad)))
print(torch.all(grad_splade_x.isclose(x.grad)))
print(torch.all(grad_splade_b.isclose(b.grad)))

