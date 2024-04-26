import torch
from torch.autograd import Function
from torch.nn import ReLU
import triton
import triton.language as tl

@triton.jit
def gradient_w_kernel(index_ptr, grad_ptr, delta_ptr, x_ptr,
                      stride_grad_dim_d, stride_grad_dim_v,
                      stride_delta_dim_b, stride_delta_dim_v,
                      stride_x_dim_b, stride_x_dim_l, stride_x_dim_d,
                      N_INDEX, D,
                      BLOCK_SIZE_D: tl.constexpr,
                      BLOCK_SIZE_INDEX: tl.constexpr,
                      GROUP_SIZE_INDEX):

    ################# L2 cache optimisation ###################
    pid = tl.program_id(axis=0)
    num_pid_index = tl.cdiv(N_INDEX, BLOCK_SIZE_INDEX)
    num_pid_d = tl.cdiv(D, BLOCK_SIZE_D)
    num_pid_in_group = GROUP_SIZE_INDEX * num_pid_d
    group_id = pid // num_pid_in_group
    first_pid_index = group_id * GROUP_SIZE_INDEX
    group_size_index = min(num_pid_index - first_pid_index, GROUP_SIZE_INDEX)
    pid_index = first_pid_index + (pid % group_size_index)
    pid_d = (pid % num_pid_in_group) // group_size_index
    ###########################################################

    offset_index = (pid_index * BLOCK_SIZE_INDEX + tl.arange(0, BLOCK_SIZE_INDEX)) % N_INDEX
    offset_d = (pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)) % D

    index_ptrs = index_ptr + offset_index

    index = tl.load(index_ptrs, mask=offset_index < N_INDEX)

    b, v, lmax = index
    for test in index:
        print(test)

    grad_ptrs = grad_ptr + (offset_d[:, None] * stride_grad_dim_d + v * stride_grad_dim_v)

    delta_ptr = delta_ptr + (b * stride_delta_dim_b + v * stride_delta_dim_v)

    x_ptrs = x_ptr + (b * stride_x_dim_b + lmax * stride_x_dim_l, offset_d[:, None] * stride_x_dim_d)

    for d in range(0, tl.cdiv(D, BLOCK_SIZE_D)):
        grad = tl.load(grad_ptrs, mask=offset_d[:, None] < D - d * BLOCK_SIZE_D, other=0.0)
        delta = tl.load(delta_ptr)
        x = tl.load(x_ptrs, mask=offset_d[None, None, :] < D - d * BLOCK_SIZE_D, other=0.0)

        grad += delta * x
        tl.store(grad_ptrs, grad, mask=offset_d[:, None] < D - d * BLOCK_SIZE_D)

######### forcement y'a des problemes #####################

def compute_gradient_w(index, delta, x):
    N_INDEX = len(index)
    D = x.shape[2]
    V = delta.shape[1]
    grad_weight = torch.zeros(D, V, dtype=torch.float64)

    grid = lambda meta: (triton.cdiv(N_INDEX, meta['BLOCK_SIZE_INDEX']) * triton.cdiv(D, meta['BLOCK_SIZE_D']), )

    gradient_w_kernel[grid](index, grad_weight, delta, x,
                            grad_weight.stride(0), grad_weight.stride(1),
                            delta.stride(0), delta.stride(1),
                            x.stride(0), x.stride(1), x.stride(2),
                            N_INDEX, D,
                            BLOCK_SIZE_D=32, BLOCK_SIZE_INDEX=1,
                            GROUP_SIZE_INDEX=8)

    return grad_weight

# def gradient_w_kernel(index_ptr, grad_ptr, delta_ptr, x_ptr,
#                       stride_index_dim_b, stride_index_dim_v,
#                       stride_grad_dim_d, stride_grad_dim_v,
#                       stride_delta_dim_b, stride_delta_dim_v,
#                       stride_x_dim_b, stride_x_dim_l, stride_x_dim_d,
#                       B, V, D,
#                       BLOCK_SIZE_B: tl.constexpr,
#                       BLOCK_SIZE_V: tl.constexpr,
#                       BLOCK_SIZE_D: tl.constexpr,
#                       GROUP_SIZE_V: tl.constexpr):
#
#     pid = tl.program_id(axis=0)
#
#     ################# L2 cache optimisation ###################
#     num_pid_v = tl.cdiv(V, BLOCK_SIZE_V)
#     num_pid_b = tl.cdiv(B, BLOCK_SIZE_B)
#
#     num_pid_in_group = GROUP_SIZE_V * num_pid_b
#     group_id = pid // num_pid_in_group
#     first_pid_v = group_id * GROUP_SIZE_V
#     group_size_v = min(num_pid_v - first_pid_v, GROUP_SIZE_V)
#     pid_v = first_pid_v + (pid % group_size_v)
#     pid_b = (pid % num_pid_in_group) // group_size_v
#     ###########################################################
#
#     offset_v = (pid_v * BLOCK_SIZE_V + tl.arange(0, BLOCK_SIZE_V)) % V
#     offset_b = (pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)) % B
#     offset_d = tl.arange(0, BLOCK_SIZE_D)
#
#     index_ptrs = index_ptr + (offset_b[:, None] * stride_index_dim_b + offset_v[None, :] * stride_index_dim_v)
#
#     pass
#
def gradient_x_kernel(index_ptr, grad_ptr, delta_ptr, x_ptr,
                      stride_index_dim_b, stride_index_dim_v,
                      stride_grad_dim_d, stride_grad_dim_v,
                      stride_delta_dim_b, stride_delta_dim_v,
                      stride_x_dim_b, stride_x_dim_l, stride_x_dim_d,
                      B, V, D,
                      BLOCK_SIZE_B: tl.constexpr,
                      BLOCK_SIZE_V: tl.constexpr,
                      BLOCK_SIZE_D: tl.constexpr,
                      GROUP_SIZE_V: tl.constexpr):

    pid = tl.program_id(axis=0)

    start_block_b = pid * BLOCK_SIZE_B

    offset_b = start_block_b + tl.arange(0, BLOCK_SIZE_B)
    offset_v = tl.arange(0, V)





class OptimReluMaxLinear(Function):
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
        x, weight, bias, effective_indice = ctx.saved_tensors
        grad_x = grad_weight = grad_bias = None

        if ctx.needs_input_grad[1]:
            grad_weight = compute_gradient_w(effective_indice, grad_outputs[0], x)

        return grad_x, grad_weight, grad_bias, None

