from typing import List

from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    def __init__(self, optimizer, step_size, last_epoch=-1, gamma=0.1):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        # ... Your Code Here ...
        self.step_size = step_size
        self.gamma = gamma

        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)
        # leave alone 

    def get_lr(self) -> List[float]:
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        # ... Your Code Here ...
        # Here's our dumb baseline implementation:
        return [base_lr * self.gamma ** (self.last_epoch // self.step_size) 
        for base_lr in self.base_lrs]
