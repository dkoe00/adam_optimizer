import math
import numpy as np
import torch

from collections import defaultdict
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

        if lr < 0 or any(0 > beta or 1 < beta for beta in betas) or eps < 0:
            raise ValueError("Invalid hyperparameters")

        super().__init__(params, defaults)
        self.state = defaultdict()
        return


    def step(self) -> None:

        for p in self.params:
            if not self.state[p]:
                self.state[p] = {
                    "exp_avg": torch.zeros_like(p),
                    "exp_avg_sq": torch.zeros_like(p),
                    "step": 0
                }
            
            grad = p.grad

        return