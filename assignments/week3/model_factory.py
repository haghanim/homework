import torch
from model import MLP


def create_model(input_dim: int, output_dim: int) -> MLP:
    """
    Create a multi-layer perceptron model.

    Arguments:
        input_dim (int): The dimension of the input data.
        output_dim (int): The dimension of the output data.
        hidden_dims (list): The dimensions of the hidden layers.

    Returns:
        MLP: The created model.

    """
    hidden_size = 32 * 2
    hidden_count = 3

    return MLP(
        input_dim,
        hidden_size,
        output_dim,
        hidden_count,
        torch.nn.ReLU,
        torch.nn.init.uniform_,
    )
