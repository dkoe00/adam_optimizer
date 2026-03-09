import pytest
import torch

import miniadam


def test_init_correct_input():
    lr = 1
    betas = (0.8, 0.7)
    eps = 1e-7
    weight_decay = 0.01

    params = []
    for i in range(5):
        params.append(torch.randn(20,i))

    test_adam = miniadam(params, lr, betas, eps, weight_decay)

    for i, group in enumerate(test_adam.param_groups):
        assert group.lr == lr
        assert group.betas == betas
        assert group.eps == eps
        assert group.weight_decay == weight_decay
        assert group.params == params[i]
        