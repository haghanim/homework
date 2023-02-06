import torch
from typing import Callable
from numpy import ndarray
import torch

class MLP(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super(MLP, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size 
        self.num_classes = num_classes 
        self.hidden_count = hidden_count 
        self.activation = activation 
        self.initializer = initializer 
        
        # define first layer 
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.initializer(self.fc1.weight)

        # define hidden layers 
        layers = [torch.nn.Linear(self.hidden_size, self.hidden_size) for i in range(0, self.hidden_count - 1)]
        self.hidden_layers = torch.nn.ModuleList(layers)

        for hidden_layer in self.hidden_layers:
            self.initializer(hidden_layer.weight)

        # define last layer
        self.fcn = torch.nn.Linear(self.hidden_size, self.num_classes)
        self.initializer(self.fcn.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        # for each layer x @ w. then repeat until you get values for each class

        # 1. first layer 
        out = self.fc1(x)
        out = self.activation(out)
        
        # 2. hidden layers
        for hidden_layer in self.hidden_layers:
            out = hidden_layer(out)
            out =self.activation(out)

        out = self.fcn(out)

        return out

