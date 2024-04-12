import torch
from torch.autograd import Function

class SpladeModel(Function):
    @staticmethod
    def forward(x, weight, bias):
        '''
        :param x: Matrix with shape (B, L, D) (Batch, sequence, embedding)
        :param weight: Matrix with shape (D, V) (embedding, vocabulary)
        :param bias: bias vector with shape (V)
        :return: the calcul of Max_on_L(x @ w + b)
        '''
        output = x @ weight
        if bias is not None:
            output += bias
        maximum, max_indice = torch.max(output, 1, keepdim=True)
        return maximum, max_indice

    @staticmethod
    def setup_context(ctx, input, output):
        x, weight, bias = input
        _, max_indice = output
        ctx.save_for_backward(x, weight, bias, max_indice)

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, weight, bias, max_indice = ctx.saved_tensors
        grad_x = grad_weight = grad_bias = None
        mask = torch.zeros_like(x @ weight)
        mask.scatter_(1, max_indice, grad_outputs[0])
        x_reduced = x[max_indice]

        if ctx.needs_input_grad[0]:
            grad_x = mask @ weight.t()

        if ctx.needs_input_grad[1]:
            grad_weight = x.t() @ mask

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_outputs[0]

        return grad_x, grad_weight, grad_bias

class SpladeModel_ReLU(Function):
    @staticmethod
    def forward(x, weight, bias):
        '''
        :param x: Matrix with shape (B, L, D) (Batch, sequence, embedding)
        :param weight: Matrix with shape (D, V) (embedding, vocabulary)
        :param bias: bias vector with shape (V)
        :return: the calcul of ReLU(x @ w + b)
        '''
        output = x @ weight
        if bias is not None:
            output += bias
        maximum = torch.max(output, torch.zeros_like(output))
        return maximum

    @staticmethod
    def setup_context(ctx, input, output):
        x, weight, bias = input
        maximum = output
        ctx.save_for_backward(x, weight, bias, maximum)

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, weight, bias, maximum = ctx.saved_tensors
        grad_x = grad_weight = grad_bias = None
        mask = torch.zeros_like(maximum)
        indice = torch.nonzero(maximum, as_tuple=True)
        mask[indice] = grad_outputs[0][indice]
        if ctx.needs_input_grad[0]:
            grad_x = mask @ weight.t()

        if ctx.needs_input_grad[1]:
            grad_weight = x.t() @ mask

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_outputs[0]

        return grad_x, grad_weight, grad_bias

