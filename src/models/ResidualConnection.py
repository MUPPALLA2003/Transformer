import torch
import torch.nn as nn
from LayerNorm import LayerNormalization

class ResidualNetwork(nn.Module):

    def __init__(self,d_model:int,dropout:float):
        super().__init__()

        self.n_embd = d_model
        self.dropout = nn.Dropout(dropout)
        self.layernorm = LayerNormalization(d_model)


    def forward(self,x:torch.Tensor,sublayer) -> torch.Tensor:

        return x + self.dropout(self.layernorm(sublayer(x)))