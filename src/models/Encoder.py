import torch
import torch.nn as nn
from Attention import Attention
from FeedForwardNN import FeedForward
from ResidualConnection import ResidualNetwork

class Encoder(nn.Module):

    def __init__(self,d_model:int,h:int,dropout:float,mask:bool,hidden_dropout:float,output_dropout:float,mlp_ratio:int):

        super().__init__()

        self.d_model = d_model
        self.attention = Attention(d_model,h,dropout,mask)
        self.mlp = FeedForward(d_model,hidden_dropout,output_dropout,mlp_ratio)
        self.residual_network = nn.ModuleList([ResidualNetwork(d_model,dropout) for i in range(2)])


    def forward(self,x:torch.Tensor):

        x = self.residual_network[0](x,self.attention)
        x = self.residual_network[1](x,self.mlp)

        return x



        

        