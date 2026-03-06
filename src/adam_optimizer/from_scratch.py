import math
import torch

from typing import Iterable

class Adam():

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 0.001,
        betas: tuple(float, float) = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.001,
    ):

        """
        initialize Adam object with given hyperparameters, sets up parameter groups, defaults, and state

        inputs:
        params: Iterable[torch.Tensor], Iterable of parameter groups,
        lr: float, desired learning rate,
        betas: tuple(float, float), update rates for exponential moving averages,
        eps: float, regularization value to avoid zero division,
        weight_decay: float, #TODO @dkoe00: document

        returns: None
        """

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        self.param_groups = []
        for group in params:
            group_dict = defaults
            group_dict[params] = group
            self.param_groups.append(group_dict)

        self.state = {}

        return

    
    def load_state_dict(self, state_dict: dict) -> None:

        """
        loads the state of the optimizer from an appropriate dict

        inputs:
        state_dict: dict, a dict containing the desired state of the optimizer

        returns:
        None
        """

        self.state = state_dict
        return


    def state_dict(self) -> dict:

        """
        exports the current state of the optimizer as a dict

        inputs:
        None

        returns:
        dict, state attribute of the optimizer at function call
        """

        return self.state


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
                #TODO @dkoenig: implement weight decay functionality
                                
                m = state["exp_avg"] * beta1 + (1 - beta1) * grad
                v = state["exp_avg_sq"] * beta2 + (1 - beta2) * grad ** 2

                state["step"] += 1
                t = state["step"]
                a = lr * (math.sqrt(1 - beta2 ** t)/(1 - beta1 ** t))

                with torch.no_grad():
                    p -= a * m / (torch.sqrt(v) + eps)
                    self.state[p]["exp_avg"] = m
                    self.state[p]["exp_avg_sq"] = v

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
            for p in group["params"]:
                if not set_to_None:
                    p.grad = torch.zeros_like(p)
                else:
                    p.grad = None

        return