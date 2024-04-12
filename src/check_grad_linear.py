import unittest
from linear_model import LinearModel
import torch

def mse(y_hat, y):
    return (torch.linalg.norm(y_hat - y)) / 2 / y.shape[0]

class MyTestCase(unittest.TestCase):
    def check_tensor(self, tensor1, tensor2):
        return torch.all(tensor1 == tensor2).item()

    def test_linear_model(self):
        x = torch.randn(10, 5, requires_grad=True, dtype=torch.float64)
        w = torch.randn(5, 1, requires_grad=True, dtype=torch.float64)
        b = torch.randn(10, 1, requires_grad=True, dtype=torch.float64)

        y = torch.randn(10, 1, requires_grad=True, dtype=torch.float64)

        self.assertEqual(torch.autograd.gradcheck(mse, (LinearModel.apply(x, w, b), y)), True)  # add assertion here


if __name__ == '__main__':

    unittest.main()
