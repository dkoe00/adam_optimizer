# miniadam

This is my minimal implementation of the optimization algorithm proposed by Kingma and Ba in their paper "Adam: A Method for Stochastic Optimization" (2015).

At the moment, please note that this is a work in progress.

The Adam class is an optimizer with the following methods and attributes:

```
class Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
```
Parameters:
- params: Iterable[torch.Tensor], all model parameters that are going to be optimized
- lr: float, learning rate used for gradient descent
- betas: tuple(float, float), update rates for the exponential moving averages
- eps: float, regularization parameter to avoid zero division
- weight_decay: float, implemented as decoupled weight decay like in AdamW

Methods:

```load_state_dict(self, state_dict: dict) -> None```:
- loads the desired state of the optimizer from a saved dict
- inputs: state_dict, a dict containing the desired state
- returns: None

```state_dict(self) -> dict```:
tbd

There is also a working training loop using an implementation of the algorithm to be found in the first_implementation notebook.