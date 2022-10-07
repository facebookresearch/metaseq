import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, List


def relu_squared(x: torch.Tensor):
    return F.relu(x).pow(2)


@torch.jit.script
def gelu(x):
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


def swiglu(x: torch.Tensor, gate: torch.Tensor):
    return F.silu(x) * gate


def geglu(x: torch.Tensor, gate: torch.Tensor):
    return gelu(x) * gate


def get_available_activation_fns() -> List:
    return [
        "relu",
        "relu_squared",
        "gelu",
        "tanh",
        "linear",
        "swiglu",
        "geglu",
    ]


class ActivationFn(nn.Module):
    def __init__(self, name, fc1_type, embed_dim, ffn_dim, **fc1_kargs):
        super().__init__()
        self.fn = self.__get_fn(name)
        self.gate = None
        if self.fn in self.__get_gated_fns():
            self.gate = fc1_type(embed_dim, ffn_dim, **fc1_kargs)

    def forward(self, fc1_in, fc1_out, model_parallel):
        if self.gate is not None:
            if model_parallel:
                g, _ = self.gate(fc1_in)
            else:
                g = self.gate(fc1_in)
            return self.fn(fc1_out, g)
        return self.fn(fc1_out)

    def __get_fn(self, name: str) -> Callable:
        """Returns the activation function corresponding to the arg passed in the run"""

        if name == "relu":
            return F.relu
        elif name == "relu_squared":
            return relu_squared
        elif name == "gelu":
            return gelu
        elif name == "tanh":
            return torch.tanh
        elif name == "linear":
            return lambda x: x
        elif name == "swiglu":
            return swiglu
        elif name == "geglu":
            return geglu
        else:
            raise RuntimeError("--activation-fn {} not supported".format(name))

    def __get_gated_fns(self) -> List:
        return [
            geglu,
            swiglu,
        ]
