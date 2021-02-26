import torch
from torch import nn


class Lambda(nn.Module):

    def __init__(self, lambda_function):
        super(Lambda, self).__init__()
        self.lambda_function = lambda_function

    def forward(self, *args):
        if len(args) == 1:
            return self.lambda_function(args[0])
        else:
            return self.lambda_function(args)


class Reshape(nn.Module):

    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, inp: torch.Tensor):
        return inp.view(self.shape)
