import unittest
import torch
from splade_model import SpladeModel


class MyTestCase(unittest.TestCase):
    def test_splade_model(self):
        batch = 32
        length = 10
        data_size = 32

        vocabulary = 20

        list_lengths = torch.randint(5, length, [batch])

        mask = torch.zeros(batch, length, dtype=torch.float64)
        for i, l in enumerate(list_lengths):
            mask[i, l:] = -torch.inf

        x = torch.randn(batch, length, data_size, requires_grad=True, dtype=torch.float64)
        w = torch.randn(data_size, vocabulary, requires_grad=True, dtype=torch.float64)
        b = torch.randn(vocabulary, requires_grad=True, dtype=torch.float64)

        self.assertEqual(torch.autograd.gradcheck(SpladeModel.apply, (x, w, b, mask), eps=1e-2, atol=0.1, rtol=1e-2), True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
