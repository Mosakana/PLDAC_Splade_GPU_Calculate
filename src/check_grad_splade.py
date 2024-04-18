import unittest
import torch
from splade_model import SpladeModel


class MyTestCase(unittest.TestCase):
    def test_splade_model(self):
        batch = 32
        length = 10
        data_size = 32

        vocabulary = 20
        list_lengths = torch.randint(0, length, [batch])

        mask = torch.ones(batch, length, dtype=torch.float64)
        for i, l in enumerate(list_lengths):
            mask[i, l:] = 0

        x = torch.randn(batch, length, data_size, requires_grad=True, dtype=torch.float64)
        w = torch.randn(data_size, vocabulary, requires_grad=True, dtype=torch.float64)
        b = torch.randn(vocabulary, requires_grad=True, dtype=torch.float64)

        self.assertEqual(torch.autograd.gradcheck(SpladeModel.apply, (x, w, b, mask)), True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
