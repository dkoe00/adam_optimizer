# adam

This is my minimal implementation of the optimization algorithm proposed by Kingma and Ba in their paper "Adam: A Method for Stochastic Optimization" (2015).

The Adam class is an optimizer inheriting from torch.optim.Optimizer. The class's constructor can be called as follows:

```
adam = Adam(params, lr, betas, eps)
```
with arguments:
- params: Iterable[torch.Tensor], all model parameters that are going to be optimized (required)
- lr: float = 0.001, learning rate used for gradient descent (optional)
- betas: tuple[float, float] = (0.9,0.999), update rates for the exponential moving averages (optional)
- eps: float = 1e-8, regularization parameter to avoid zero division (optional)

The other methods, particularly `Adam.step`, follow the standard contract for pytorch optimizers, minus optional arguments listed in the torch.optim docs but not above.

Two training loops for a GPT-like transformer model on the tiny shakespeare dataset (as a hommage to Andrej Karpathy's excellent [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) lecture series) can be found in the training_demo notebook, one using my implementation of the optimizer and one using the standard torch.optim.Adam optimizer.
The notebook demonstrates that model training with our implementation works just as well as using the official pytorch implementation.
The checkpoints directory contains the trained models for reference.

There is also a working training loop for an extremely basic model using an implementation of the algorithm to be found in the first_implementation notebook.
Please note that this is purely for educational purposes and does not follow best practices for model training with pytorch or the standard optimizer contract.