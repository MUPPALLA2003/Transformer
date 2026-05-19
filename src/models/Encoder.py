import torch
import torch.nn as nn
from Attention import Attention
from FeedForwardNN import FeedForward
from ResidualConnection import ResidualNetwork

class Encoder(nn.Module):

    def __init__(self,d_model:int,max_seq_len_src:int,h:int,dropout:float,causal_mask:bool,hidden_dropout:float,output_dropout:float,mlp_ratio:int):

        super().__init__()

        self.d_model = d_model
        self.attention = Attention(d_model,max_seq_len_src,h,dropout,causal_mask)
        self.mlp = FeedForward(d_model,hidden_dropout,output_dropout,mlp_ratio)
        self.residual_network = nn.ModuleList([ResidualNetwork(d_model,dropout) for _ in range(2)])


    def forward(self,x:torch.Tensor,src_mask = None):

        x = self.residual_network[0](x,self.attention)
        x = self.residual_network[1](x,self.mlp)

        return x



        

        