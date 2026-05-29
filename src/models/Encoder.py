import torch
import torch.nn as nn
from Attention import Attention
from FeedForwardNN import FeedForward
from ResidualConnection import ResidualNetwork

class Encoder(nn.Module):

    def __init__(self,d_model:int,max_seq_len_src:int,h:int,dropout:float,hidden_dropout:float,output_dropout:float,mlp_ratio:int):

        super().__init__()

        self.d_model = d_model
        self.attention = Attention(d_model,max_seq_len_src,h,dropout,causal=False)
        self.mlp = FeedForward(d_model,hidden_dropout,output_dropout,mlp_ratio)
        self.residual_network = nn.ModuleList([ResidualNetwork(d_model,dropout) for _ in range(2)])


    def forward(self,src:torch.Tensor,src_mask = None):

        src = self.residual_network[0](src,lambda src:self.attention(src,src,src,src_mask))
        src = self.residual_network[1](src,self.mlp)

        return src



        

        