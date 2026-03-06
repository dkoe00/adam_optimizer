import math
import torch

from typing import Iterable

class Adam():

    def __init__(
        self,
        params: Iterable[torch.Tensor | dict],
        lr: float = 0.001,
        betas: tuple(float, float) = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.001,
    ):

        """
        initialize Adam object with given hyperparameters

        inputs:
        params: Iterable[torch.Tensor | dict], Iterable of parameter groups,
        lr: float, desired learning rate,
        betas: tuple(float, float), update rates for exponential moving averages,
        eps: float, regularization value to avoid zero division,
        weight_decay: float, #TODO @dkoe00: document
        """

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        self.param_groups = []
        for group in params:
            group_dict = {}
            self.param_groups.append()