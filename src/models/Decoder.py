import torch
import torch.nn as nn
from Attention import Attention
from ResidualConnection import ResidualNetwork
from FeedForwardNN import FeedForward

class Decoder(nn.Module):

    def __init__(self,d_model:int,h:int,dropout:float,hidden_dropout:float,output_dropout:float,mlp_ratio:int,max_seq_len_src:int,max_seq_len_tgt:int):

        super().__init__()

        self.causal_attention = Attention(d_model,max_seq_len_src,h,dropout,causal = True)
        self.mlp = FeedForward(d_model,hidden_dropout,output_dropout,mlp_ratio)
        self.cross_attention = Attention(d_model,max_seq_len_tgt,h,dropout,causal = False)
        self.residual_network = nn.ModuleList(ResidualNetwork(d_model,dropout) for _ in range(3))

    def forward(self,src:torch.Tensor,tgt:torch.Tensor,src_mask = None,tgt_mask = None):

        x = self.residual_network[0](x,self.causal_attention)
        x = self.residual_network[1](x,self.cross_attention)
        x = self.residual_network[2](x,self.mlp)

        return x