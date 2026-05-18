import torch
import torch.nn as nn
import math

class Attention(nn.Module):

    def __init__(self,d_model:int,max_Seq_len:int,h:int,dropout:float,causal:bool):

        super().__init__()

        assert d_model % h == 0

        self.d_model = d_model
        self.h = h
        self.d_k = d_model//h
        self.dropout = nn.Dropout(dropout)
        self.causal = causal
        self.Q = nn.Linear(d_model,d_model,bias = False)
        self.K = nn.Linear(d_model,d_model,bias = False)
        self.V = nn.Linear(d_model,d_model,bias = False)
        self.W = nn.Linear(d_model,d_model,bias = False)

        if causal:

            causal_mask = torch.triu(torch.ones(max_Seq_len,max_Seq_len), diagonal=1).bool()
            self.register_buffer("causal_mask", causal_mask)

    def attention(self,q,k,v,dropout=None,key_padding_mask=None):

        d_k = q.size(-1)
        T_q, T_kv = q.size(-2), k.size(-2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
        
        if self.causal:

            scores = scores.masked_fill(self.causal_mask[:T_q, :T_kv], float('-inf'))
            
        if key_padding_mask is not None:

            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float('-inf'))    

        attention_probs = scores.softmax(dim=-1)

        if dropout is not None:

            attention_probs = dropout(attention_probs)

        return attention_probs @ v


    def forward(self,query,key,value,key_padding_mask=None):

        B,  T_q, C = query.shape
        _, T_kv, _ = key.shape
        q = self.Q(query)
        k = self.K(key)
        v = self.V(value)

        q = q.view(B, T_q, self.h, self.d_k).transpose(1, 2)
        k = k.view(B, T_kv, self.h, self.d_k).transpose(1, 2)
        v = v.view(B, T_kv, self.h, self.d_k).transpose(1, 2)
        out = self.attention(q,k,v,self.mask,self.dropout,key_padding_mask)
        out = out.transpose(1, 2).contiguous().view(B, T_q, C)

        return self.W(out)    


