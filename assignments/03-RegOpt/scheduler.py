from typing import List

from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    A custom learning rate scheduler that extends the PyTorch _LRScheduler class.
    """

    def __init__(
        self,
        optimizer,
        step_size,
        num_epochs,
        initial_learning_rate,
        initial_weight_decay,
        batch_size,
        last_epoch=-1,
        gamma=0.1,
    ):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        # ... Your Code Here ...
        self.step_size = step_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.initial_learning_rate = initial_learning_rate
        self.initial_weight_decay = initial_weight_decay

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
        # return [i for i in self.base_lrs]

        learning_rates = []
        lr = self.initial_learning_rate

        # DID NOT HELP
        # # lr = lr / (self.batch_size ** 0.5)

        for group in self.optimizer.param_groups:

            # DID NOT HELP
            # if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            #     lr = lr
            # lr = (self.last_epoch // self.step_size) * lr

            lr = self.gamma * lr
            learning_rates.append(lr)

        # Return the learning rate
        return learning_rates
