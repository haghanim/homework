from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, ToTensor


class CONFIG:
    # batch_size = 64
    # num_epochs = 10
    # initial_learning_rate = 0.01
    # initial_weight_decay = 0

    batch_size = 64
    num_epochs = 10
    initial_learning_rate = 0.01
    initial_weight_decay = 0

    step_size = 10
    gamma = 0.1

    # ?? 128 batch_size -> higher test acc (just right, more is worse)??

    lrs_kwargs = {
        # You can pass arguments to the learning rate scheduler
        # constructor here.
        "gamma": gamma,
        "num_epochs": num_epochs,
        "initial_learning_rate": initial_learning_rate,
        "step_size": step_size,
        "batch_size": batch_size,
        # 'initial_weight_decay': initial_weight_decay
    }

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(
        model.parameters(),
        lr=CONFIG.initial_learning_rate,
        weight_decay=CONFIG.initial_weight_decay,
    )

    transforms = Compose(
        [
            ToTensor(),
        ]
    )
