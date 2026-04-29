import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self,d_model:int,vocabulary_size:int):

        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocabulary_size,d_model,dtype = torch.float32)

    def forward(self,x:torch.Tensor):

        x = self.embedding(x) * math.sqrt(self.d_model)    