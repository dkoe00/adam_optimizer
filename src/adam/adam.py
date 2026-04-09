import math
import torch

from typing import Iterable, Tuple

class Adam(torch.optim.Optimizer):

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
    ):

        """
        initialize Adam object with given hyperparameters, sets up parameter groups, defaults, and state

        inputs:
        params: Iterable[torch.Tensor], Iterable of parameter groups,
        lr: float, desired learning rate,
        betas: tuple[float, float], update rates for exponential moving averages,
        eps: float, regularization value to avoid zero division,
        weight_decay: float, parameter for decoupled weight decay like in AdamW

        returns: None
        """

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        super().__init__(params, defaults)

        return
    

    def step(self) -> None:

        """
        perform one step of the optimizer

        inputs:
        None

        returns:
        None
        """
        
        for group in self.param_groups:
            for param in group:
                state = self.state[param]
                if not state:
                    self.state[param] = {
                        "step": 1,
                        "exp_avg": torch.zeros_like(param),
                        "exp_avg_sq": torch.zeros_like(param),
                    }
                    state = self.state[param]

                grad = param.grad
                if not grad:
                    continue

                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                                
                m = state["exp_avg"] * beta1 + (1 - beta1) * grad
                v = state["exp_avg_sq"] * beta2 + (1 - beta2) * grad ** 2

                state["step"] += 1
                t = state["step"]
                a = lr * (math.sqrt(1 - beta2 ** t)/(1 - beta1 ** t))

                with torch.no_grad():
                    param -= lr * weight_decay * param
                    param -= a * m / (torch.sqrt(v) + eps)
                    self.state[param]["exp_avg"] = m
                    self.state[param]["exp_avg_sq"] = v

        return


    def zero_grad(self, set_to_None: bool = False) -> None:

        """
        sets gradients of all parameters to zero or None

        inputs:
        set_to_None: bool, decide whether to replace all elements in the grads with zero or make the entire grad None

        returns:
        None
        """

        for group in self.param_groups:
            for param in group["params"]:
                if not set_to_None:
                    param.grad = torch.zeros_like(param)
                else:
                    param.grad = None

        return