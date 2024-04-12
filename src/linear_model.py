from torch.autograd import Function

class LinearModel(Function):
    @staticmethod
    def forward(x, weight, bias):
        output = x.mm(weight)
        if bias is not None:
            output += bias
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, weight, bias = inputs
        ctx.save_for_backward(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        grad_x = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_x = grad_output.mm(weight.t())
        if ctx.needs_input_grad[1]:
            grad_weight = x.t().mm(grad_output)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output

        return grad_x, grad_weight, grad_bias


