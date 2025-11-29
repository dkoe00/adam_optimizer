import math
import numpy as np
import torch

from typing import Callable

class Adam():

    def __init__(
            self, 
            func: Callable[[torch.Tensor], float], 
            alpha: float = 0.001, 
            beta_1: float = 0.9, 
            beta_2: float = 0.999, 
            epsilon: float = 10e-8, 
            max_iters: int = 10000,
            grad_tol: float = 1e-4
        ) -> None:
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.func = func
        self.max_iters = max_iters
        self.grad_tol = grad_tol
        return


    def step(self, t: int, current_theta: torch.Tensor, m: torch.Tensor, v: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor, float):

        # zero the gradients
        current_theta.grad = None

        # compute the gradients of f w.r.t. the current parameters
        # forward pass
        objective = self.func(current_theta)
        # backward pass
        objective.backward()
        g = current_theta.grad
        g_norm = g.max().item()
    
        # compute stochastic moments
        m = self.beta_1 * m + (1 - self.beta_1) * g
        v = self.beta_2 * v + (1 - self.beta_2) * g ** 2
        a = self.alpha * (math.sqrt(1 - self.beta_2 ** t)/(1 - self.beta_1 ** t))

        # update parameters
        with torch.no_grad():
            next_theta = current_theta - a * m / (torch.sqrt(v) + self.epsilon)

        return next_theta, m, v, g_norm
        
        
    def optimize(self, theta: np.ndarray) -> np.ndarray:

        m = torch.zeros(theta.shape, dtype=torch.float)
        v = torch.zeros(theta.shape, dtype=torch.float)

        theta = np.asarray(theta, dtype=np.float32)
        last_theta = torch.from_numpy(theta)
        next_theta = torch.from_numpy(theta) + 2 * self.epsilon

        t = 0
        g_norm = 1.0

        # until the parameters converge
        while g_norm > self.grad_tol:

            t += 1
            last_theta = next_theta.detach().requires_grad_(True)
            next_theta, m, v, g_norm = self.step(t, last_theta, m, v)

            if t > self.max_iters:
                break

        return next_theta.detach().numpy()