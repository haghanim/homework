from typing import List

from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    A custom learning rate scheduler that extends the PyTorch _LRScheduler class.
    """

    def __init__(self, optimizer, step_size, last_epoch=-1, gamma=0.1):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        # ... Your Code Here ...
        self.step_size = step_size
        self.gamma = gamma

        # batch_size,num_epochs,initial_learning_rate,last_epoch

        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)
        # leave alone

    def get_lr(self) -> List[float]:
        """
        Get the current learning rate for each parameter group.

        Returns:
            A list of floats representing the current learning rate for each
            parameter group in the optimizer.
        """

        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        # ... Your Code Here ...
        # Here's our dumb baseline implementation:
        return [i for i in self.base_lrs]
