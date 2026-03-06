import math
import torch

from typing import Iterable

class Adam(torch.optim.Optimizer):

    def __init__(
            self, 
            params: Iterable[torch.Tensor | dict],
            lr: float = 0.001, 
            betas: tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8, 
            weight_decay: float = 0,
        ) -> None:

        """
        initialize Adam object with the given hyperparameters

        inputs:
        params: Iterable[torch.Tensor | dict], existing parameters
        lr: float, desired learning rate
        betas: tuple[float, float], the update rates of the exponential moving averages
        eps: float, regularization parameter to avoid zero division errors
        weight_decay: float, #TODO @dkoe00: document

        returns:
        None
        """

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }

        if lr <= 0:
            raise ValueError("Invalid hyperparameters: lr must be > 0")
        if any(0 >= beta or 1 <= beta for beta in betas):
            raise ValueError("Invalid hyperparameters: betas must be > 0 and < 1")
        if eps <= 0:
            raise ValueError("Invalid hyperparameters: eps must be >0")
        #TODO @dkoe00: implement weight_decay check

        super().__init__(params, defaults)
        return


    def load_state_dict(self, state_dict: StateDict) -> None:

        """
        load Adam object's state from a StateDict

        inputs:
        state_dict: StateDict, contains the desired object state

        returns:
        None
        """

        return super().load_state_dict(state_dict)


    def state_dict(self) -> StateDict:
        return super().state_dict()
    

    def step(self) -> None:

        """
        perform one optimization step

        inputs:
        None

        returns:
        None
        """

        #TODO @dkoe00: add use of weight_decay parameter

        for group in self.param_groups:
            for p in group["params"]:
                grad = p.grad

                if grad is None or grad.is_sparse:
                    continue
                
                state = self.state[p]
                if len(state) == 0:
                    self.state[p] = {
                        "exp_avg": torch.zeros_like(p),
                        "exp_avg_sq": torch.zeros_like(p),
                        "step": 0
                    }
                    state = self.state[p]

                lr = group["lr"]
                beta_1, beta_2 = group["betas"]
                eps = group["eps"]
                
                m = state["exp_avg"] * beta_1 + (1 - beta_1) * grad
                v = state["exp_avg_sq"] * beta_2 + (1 - beta_2) * grad ** 2

                state["step"] += 1
                t = state["step"]
                a = lr * (math.sqrt(1 - beta_2 ** t)/(1 - beta_1 ** t))

                with torch.no_grad():
                    p -= a * m / (torch.sqrt(v) + eps)
                    self.state[p]["exp_avg"] = m
                    self.state[p]["exp_avg_sq"] = v

        return


    def zero_grad(self, set_to_none: bool = True) -> None:

        """
        zero the gradients of the parameters

        inputs:
        set_to_none: bool, flag to set gradients to None rather than actual zeros

        returns:
        None
        """

        return super().zero_grad(set_to_none)