from functools import partial
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _lr_lambda(step: int,d_model: int,warmup_steps: int) -> float:
  
    if d_model <= 0:

        raise ValueError("d_model must be positive.")

    if warmup_steps <= 0:

        raise ValueError("warmup_steps must be positive.")

    step = max(step, 1)

    return (d_model ** -0.5* min(step ** -0.5,step * warmup_steps ** -1.5))


class TransformerLRScheduler(LambdaLR):
    
    def __init__(self,optimizer:Optimizer,d_model:int,warmup_steps:int,last_epoch:int = -1):

        self.d_model = d_model
        self.warmup_steps = warmup_steps

        lr_lambda = partial(_lr_lambda,d_model=d_model,warmup_steps=warmup_steps)

        super().__init__(optimizer,lr_lambda=lr_lambda,last_epoch=last_epoch)

    def __repr__(self) -> str:

        return (f"{self.__class__.__name__}("f"d_model={self.d_model}, "f"warmup_steps={self.warmup_steps})")