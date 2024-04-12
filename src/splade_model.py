import torch
from torch.autograd import Function
from torch.nn import ReLU

class SpladeModel(Function):
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
        output += mask.reshape(*mask.shape, 1)

        relu = ReLU().double()
        maximum, max_indice = torch.max(output, 1)
        max_indice = torch.where(maximum >= 0, max_indice, -max_indice)

        return relu(maximum), max_indice

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        x, weight, bias, _ = inputs
        _, max_indice = outputs

        ctx.save_for_backward(x, weight, bias, max_indice)

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, weight, bias, max_indice = ctx.saved_tensors
        grad_x = grad_weight = grad_bias = None

        B, L, D = x.shape
        V = weight.shape[1]


        if ctx.needs_input_grad[0]:
            grad_x = torch.zeros_like(x, dtype=torch.float64)
            for b in range(B):
                for v in range(V):
                    if max_indice[b, v] >= 0:
                        grad_x[b, max_indice[b, v]] += grad_outputs[0][b, v] * weight[:, v]

        if ctx.needs_input_grad[1]:
            #########################  version 1  ##################################
            # list_grad = []
            # for v in range(V):
            #     grads = torch.zeros(D)
            #     for b in range(B):
            #         if max_indice[b, v] >= 0:
            #             grads += grad_outputs[0][b, v] * x[b, max_indice[b, v]]
            #
            #     list_grad.append(grads)
            #
            # grad_weight = torch.cat(list_grad).reshape(D, V)
            ########################################################################

            #########################  version 2  ##################################

            grad_weight = torch.zeros_like(weight, dtype=torch.float64)
            for b in range(B):
                for v in range(V):
                    if max_indice[b, v] >= 0:
                        grad_weight[:, v] += grad_outputs[0][b, v] * x[b, max_indice[b, v]]

            ########################################################################

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = torch.zeros_like(bias, dtype=torch.float64)
            for b in range(B):
                for v in range(V):
                    if max_indice[b, v] >= 0:
                        grad_bias[v] += grad_outputs[0][b, v] * 1

        return grad_x, grad_weight, grad_bias, None




