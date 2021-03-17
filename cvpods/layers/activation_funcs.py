import torch
import torch.nn as nn


# Ref:
# https://medium.com/the-artificial-impostor/more-memory-efficient-swish-activation-function-e07c22c12a76
class SwishImplementation(torch.autograd.Function):
    """
    Swish activation function memory-efficient implementation.

    This implementation explicitly processes the gradient, it keeps a copy of the input tensor,
    and uses it to calculate the gradient during the back-propagation phase.
    """
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    """
    Implement the Swish activation function.
    See: https://arxiv.org/abs/1710.05941 for more details.
    """
    def forward(self, x):
        return x * torch.sigmoid(x)
