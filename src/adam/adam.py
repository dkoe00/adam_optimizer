import math
import torch

from typing import Callable, Iterable, Tuple

class Adam(torch.optim.Optimizer):

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):

        """
        initialize Adam object with given hyperparameters, sets up parameter groups, defaults, and state

        inputs:
        params: Iterable[torch.Tensor], Iterable of parameter groups,
        lr: float, desired learning rate,
        betas: tuple[float, float], update rates for exponential moving averages,
        eps: float, regularization value to avoid zero division,

        returns: None
        """

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
        }

        super().__init__(params, defaults)

        return
    

    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:

        """
        perform one step of the optimizer

        inputs:
        closure: Callable, optional reevaluation of the model

        returns:
        None
        """

        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for param in group["params"]:

                grad = param.grad
                if grad is None:
                    continue

                if grad.is_sparse:
                    raise RuntimeError("Sparse gradients not supported. Please use a dedicated optimizer.")

                if self.state[param] == {}:
                    self.state[param] = {
                        "step": 0,
                        "exp_avg": torch.zeros_like(param),
                        "exp_avg_sq": torch.zeros_like(param),
                    }
                state = self.state[param]

                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                                
                m = state["exp_avg"] * beta1 + (1 - beta1) * grad
                v = state["exp_avg_sq"] * beta2 + (1 - beta2) * grad ** 2

                state["step"] += 1
                t = state["step"]
                a = lr * (math.sqrt(1 - beta2 ** t)/(1 - beta1 ** t))

                with torch.no_grad():
                    param -= a * m / (torch.sqrt(v) + eps)
                    self.state[param]["exp_avg"] = m
                    self.state[param]["exp_avg_sq"] = v

        return loss


    def zero_grad(self, set_to_none: bool = False) -> None:

        """
        sets gradients of all parameters to zero or None

        inputs:
        set_to_none: bool, decide whether to replace all elements in the grads with zero or make the entire grad None

        returns:
        None
        """

        for group in self.param_groups:
            for param in group["params"]:
                if not set_to_none:
                    if param.grad is not None:
                        param.grad.detach_()
                        param.grad.zero_()
                else:
                    param.grad = None

        return