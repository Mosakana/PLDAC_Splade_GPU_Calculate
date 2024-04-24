import torch
from torch.autograd import Function
from torch.nn import ReLU
import triton
import triton.language as tl

class OptimReluMaxLinear(Function):
    def gradient_x_kernel(self, index_ptr, output_ptr, output_row_stride, BLOCK_SIZE:tl.constexpr):
        # divide on B
        pid = tl.program_id(axis=0)







    @staticmethod
    def forward(x, weight, bias, mask):
        '''
        :param x: Matrix with shape (B, L, D) (Batch, sequence, embedding)
        :param weight: Matrix with shape (D, V) (embedding, vocabulary)
        :param bias: bias vector with shape (V)
        :return: the calcul of ReLU(Max_on_L(x @ w + b))
        '''

        output = x @ weight
        if bias is not None:
            output += bias.reshape(1, 1, bias.shape[0])

        mask = torch.where(mask == 1, 0, -torch.inf)
        output += mask.reshape(*mask.shape, 1)

        relu = ReLU()
        maximum, max_indice = torch.max(output, 1)
        result = relu(maximum)

        indice_not_zero = torch.nonzero(result, as_tuple=True)

        effective_indice = []

        for b, v in zip(*indice_not_zero):
            effective_indice.append((b, v, max_indice[b, v]))

        return result, torch.tensor(effective_indice)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        x, weight, bias, _ = inputs
        _, effective_indice = outputs

        ctx.save_for_backward(x, weight, bias, effective_indice)

    @staticmethod
    def backward(ctx, *grad_outputs):

