import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self,embed:int,vocab_size:int):

        super().__init__()

        self.embed = embed
        self.embedding = nn.Embedding(vocab_size,embed,dtype = torch.float32)

    def forward(self,x:torch.Tensor):

        x = self.embedding(x) * math.sqrt(self.embed)    