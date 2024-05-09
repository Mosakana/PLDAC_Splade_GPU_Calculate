import torch
from optim_relu_max_linear import OptimReluMaxLinear
import logging

def test_optim_splade_model():
    batch = 32
    length = 10
    data_size = 32

    vocabulary = 20
    list_lengths = torch.randint(0, length, [batch])

    mask = torch.ones(batch, length, dtype=torch.float64, device='cuda')
    for i, l in enumerate(list_lengths):
        mask[i, l:] = 0

    x = torch.randn(batch, length, data_size, requires_grad=True, dtype=torch.float64, device='cuda')
    w = torch.randn(data_size, vocabulary, requires_grad=True, dtype=torch.float64, device='cuda')
    b = torch.randn(vocabulary, requires_grad=True, dtype=torch.float64, device='cuda')
    torch.autograd.set_detect_anomaly(True)


    print(f'The grad calculation is correct : {torch.autograd.gradcheck(OptimReluMaxLinear.apply, (x, w, b, mask))}')


if __name__ == '__main__':
    test_optim_splade_model()
