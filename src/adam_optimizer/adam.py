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
        ) -> None:
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
        }

        if lr <= 0 or any(0 >= beta or 1 <= beta for beta in betas) or eps <= 0:
            raise ValueError("Invalid hyperparameters")

        super().__init__(params, defaults)
        return


    def step(self) -> None:

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    self.state[p] = {
                        "exp_avg": torch.zeros_like(p),
                        "exp_avg_sq": torch.zeros_like(p),
                        "step": 0
                    }

                if p.grad is None:
                    continue

                self.state[p]["step"] += 1
                
                lr = group["lr"]
                beta_1, beta_2 = group["betas"]
                eps = group["eps"]
                
                grad = p.grad
                m = self.state[p]["exp_avg"] * beta_1 + (1 - beta_1) * grad
                v = self.state[p]["exp_avg_sq"] * beta_2 + (1 - beta_2) * grad ** 2

                t = self.state[p]["step"]
                a = lr * (math.sqrt(1 - beta_2 ** t)/(1 - beta_1 ** t))

                with torch.no_grad():
                    p -= a * m / (torch.sqrt(v) + eps)
                    self.state[p]["exp_avg"] = m
                    self.state[p]["exp_avg_sq"] = v

        return