import torch
from torch.autograd import Function
from torch.nn import ReLU
import triton
import triton.language as tl

@triton.jit
def gradient_w_kernel(index_ptr, index_mask_ptr, grad_ptr, delta_ptr, x_ptr,
                      D, V, MAX_N_INDEX,
                      stride_index_dim0, stride_index_dim1, stride_index_dim2,
                      stride_grad_d, stride_grad_v,
                      stride_delta_b, stride_delta_v,
                      stride_x_b, stride_x_l, stride_x_d,
                      BLOCK_SIZE_V: tl.constexpr):

    pid = tl.program_id(axis=1)

    start_point = pid * BLOCK_SIZE_V

    offsets = start_point + tl.arange(0, BLOCK_SIZE_V)
    index_mask = tl.load(index_mask_ptr + offsets, mask=offsets < V, other=0)

    for n in range(MAX_N_INDEX):
        b = tl.load(index_ptr + offsets * stride_index_dim0 + n * stride_index_dim1 + 0 * stride_index_dim2, mask=(n < index_mask), other=0)
        v = tl.load(index_ptr + offsets * stride_index_dim0 + n * stride_index_dim1 + 1 * stride_index_dim2, mask=(n < index_mask), other=0)
        lmax = tl.load(index_ptr + offsets * stride_index_dim0 + n * stride_index_dim1 + 2 * stride_index_dim2, mask=(n < index_mask), other=0)

        delta = tl.load(delta_ptr + (b * stride_delta_b + v * stride_delta_v), mask=(n < index_mask))

        for d in range(D):
            x = tl.load(x_ptr + (b * stride_x_b + lmax * stride_x_l + d * stride_x_d), mask=(n < index_mask))
            grad = tl.load(grad_ptr + (d * stride_grad_d + v * stride_grad_v), mask=(n < index_mask))

            grad += delta * x

            tl.store(grad_ptr + (d * stride_grad_d + v * stride_grad_v), grad, mask=(n < index_mask))

def compute_gradient_w(effective_indice, mask_effective_indice, delta, x, grad_weight):
    D, V = grad_weight.shape

    BLOCK_SIZE_V = triton.next_power_of_2(V)
    num_warps = 4
    if BLOCK_SIZE_V >= 2048:
        num_warps = 8
    if BLOCK_SIZE_V >= 4096:
        num_warps = 16

    grid = lambda meta: (triton.cdiv(V, meta['BLOCK_SIZE_V']), )
    gradient_w_kernel[grid](effective_indice, mask_effective_indice, grad_weight, delta, x,
                            D, V, effective_indice.shape[1],
                            effective_indice.stride(0), effective_indice.stride(1), effective_indice.stride(2),
                            grad_weight.stride(0), grad_weight.stride(1),
                            delta.stride(0), delta.stride(1),
                            x.stride(0), x.stride(1), x.stride(2),
                            num_warps=num_warps,
                            BLOCK_SIZE_V=BLOCK_SIZE_V)

    torch.cuda.empty_cache()

    return grad_weight.clone().detach().requires_grad_(True)


@triton.jit
def gradient_x_kernel(index_ptr, grad_ptr, delta_ptr, w_ptr,
                      D, N_INDEX,
                      stride_index_dim0, stride_index_dim1,
                      stride_grad_b, stride_grad_lmax, stride_grad_d,
                      stride_delta_b, stride_delta_v,
                      stride_w_d, stride_w_v,
                      BLOCK_SIZE_D: tl.constexpr):

    pid = tl.program_id(axis=0)

    start_point = pid * BLOCK_SIZE_D

    offsets = start_point + tl.arange(0, BLOCK_SIZE_D)

    mask = offsets < D

    for i in range(N_INDEX):
        b = tl.load(index_ptr + (i * stride_index_dim0) + 0 * stride_index_dim1)
        v = tl.load(index_ptr + (i * stride_index_dim0) + 1 * stride_index_dim1)
        lmax = tl.load(index_ptr + (i * stride_index_dim0) + 2 * stride_index_dim1)

        delta = tl.load(delta_ptr + (b * stride_delta_b + v * stride_delta_v))
        w = tl.load(w_ptr + (offsets * stride_w_d + v * stride_w_v), mask=mask)

        grad = tl.load(grad_ptr + (b * stride_grad_b + lmax * stride_grad_lmax + offsets * stride_grad_d), mask=mask)

        grad += delta * w

        tl.store(grad_ptr + (b * stride_grad_b + lmax * stride_grad_lmax + offsets * stride_grad_d), grad, mask=mask)



def compute_gradient_x(index, delta, w, grad_x):
    N_INDEX = index.shape[0]
    D = w.shape[0]

    BLOCK_SIZE_D = triton.next_power_of_2(D)
    num_warps = 4
    if BLOCK_SIZE_D >= 2048:
        num_warps = 8
    if BLOCK_SIZE_D >= 4096:
        num_warps = 16

    grid = lambda meta: (triton.cdiv(D, meta['BLOCK_SIZE_D']), )
    gradient_x_kernel[grid](index, grad_x, delta, w,
                            D, N_INDEX,
                            index.stride(0), index.stride(1),
                            grad_x.stride(0), grad_x.stride(1), grad_x.stride(2),
                            delta.stride(0), delta.stride(1),
                            w.stride(0), w.stride(1),
                            num_warps=num_warps,
                            BLOCK_SIZE_D=BLOCK_SIZE_D)

    torch.cuda.empty_cache()

    return grad_x.clone().detach().requires_grad_(True)


class OptimReluMaxLinearV(Function):
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

        list_triplet = []

        for b, v in zip(*indice_not_zero):
            list_triplet.append((b, v, max_indice[b, v]))

        tensor_triplet = torch.tensor(list_triplet, device='cuda')

        unique_v = torch.unique(tensor_triplet[:, 1])

        list_tensor_indice = [tensor_triplet[tensor_triplet[:, 1] == v] for v in unique_v]

        mask_effective_indice = torch.tensor([len(t) for t in list_tensor_indice], device='cuda')

        max_shape = max([t.shape for t in list_tensor_indice])

        padded_tensors = [torch.nn.functional.pad(t, (0, max_shape[1] - t.shape[1], 0, max_shape[0] - t.shape[0]))
                          for t in
                          list_tensor_indice]

        effective_indice = torch.stack(padded_tensors)

        return result, tensor_triplet, effective_indice, mask_effective_indice

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        x, weight, bias, _ = inputs
        _, list_triplet, effective_indice, mask_effective_indice = outputs

        ctx.save_for_backward(x, weight, bias, list_triplet, effective_indice, mask_effective_indice)

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, weight, bias, list_triplet, effective_indice, mask_effective_indice = ctx.saved_tensors
        grad_x = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_x = compute_gradient_x(list_triplet, grad_outputs[0].clone(), weight, torch.zeros_like(x))

        if ctx.needs_input_grad[1]:
            grad_weight = compute_gradient_w(effective_indice, mask_effective_indice, grad_outputs[0].clone(), x, torch.zeros_like(weight))

        if ctx.needs_input_grad[2]:
            index = torch.zeros_like(grad_outputs[0], device='cuda', dtype=torch.int64)
            index[list_triplet[:, 0], list_triplet[:, 1]] = 1
            grad_bias = torch.zeros((2, grad_outputs[0].shape[1]), device='cuda',
                                    dtype=grad_outputs[0].dtype).scatter_add_(dim=0, index=index, src=grad_outputs[0])[1, :]

        return grad_x, grad_weight, grad_bias, None