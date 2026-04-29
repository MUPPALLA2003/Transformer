import torch
import torch.nn as nn
import math

class Attention(nn.Module):

    def __init__(self,d_model:int,h:int,dropout:float,mask:bool):

        super().__init__()

        assert d_model % h == 0

        self.d_model = d_model
        self.h = h
        self.d_k = d_model//h
        self.dropout = nn.Dropout(dropout)
        self.mask = mask
        self.Q = nn.Linear(d_model,d_model,bias = False)
        self.K = nn.Linear(d_model,d_model,bias = False)
        self.V = nn.Linear(d_model,d_model,bias = False)
        self.W = nn.Linear(d_model,d_model,bias = False)

    def attention(self,q,k,v,mask,dropout):

        self.d_k = q.size(-1)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask:

            T = q.size(-2)
            mask = torch.triu(torch.ones(T, T, device=q.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))

        attention_probs = scores.softmax(dim=-1)

        if dropout is not None:

            attention_probs = dropout(attention_probs)

        return attention_probs @ v


    def forward(self, x):

        B,T,C = x.shape
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        q = q.view(B, T, self.h, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.h, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.h, self.d_k).transpose(1, 2)
        out = self.attention(q,k,v,self.mask,self.dropout)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.W(out)    


