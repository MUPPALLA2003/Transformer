import torch
import torch.nn as nn

class FeedForward(nn.Module):
  
    def __init__(self,d_model:int,hidden_dropout:float,output_dropout:float,mlp_ratio:float):

        super().__init__()
        
        hidden_size = d_model * mlp_ratio
        self.hidden_layer = nn.Linear(d_model,d_model * hidden_size)
        self.activation = nn.GELU()
        self.intermediate_dropout = nn.Dropout(hidden_dropout)
        self.output_layer = nn.Linear(hidden_size,d_model)
        self.output_dropout = nn.Dropout(output_dropout)

    def forward(self,x:torch.Tensor):

        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.intermediate_dropout(x)
        x = self.output_layer(x)
        x = self.output_dropout(x)

        return x