import pytest
import torch

from adam import Adam


test_params = [torch.randn(10,i) for i in range(5)]
test_adam = Adam(test_params)

def simulate_gradients(params: torch.Tensor) -> None:
    params.grad == torch.randn_like(params)
    return


def test_before_first_step():
    assert len(test_adam.state) == 0


def test_after_steps():
    for group in test_adam.param_groups:
        simulate_gradients(group)

    test_adam.step()

    state_dict = test_adam.state_dict()

    #TODO @dkoe00: add actual tests