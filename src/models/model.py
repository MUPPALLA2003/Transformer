import torch
import torch.nn as nn
from InputEmbed import InputEmbeddings
from PositionalEncoding import PositionalEncoding
from LayerNorm import LayerNormalization
from Encoder import Encoder
from Decoder import Decoder
from Projection import ProjectionLayer
from typing import Optional
class Transformer(nn.Module):

    def __init__(self, d_model: int, src_vocabulary_size: int, tgt_vocabulary_size: int,max_seq_len_src:int,max_seq_len_tgt:int,h: int, dropout: float,hidden_dropout: float, output_dropout: float,mlp_ratio: int, N: int):

        super().__init__()

        self.src_embed = InputEmbeddings(d_model, src_vocabulary_size)
        self.tgt_embed = InputEmbeddings(d_model, tgt_vocabulary_size)
        self.wpe_src  = PositionalEncoding(d_model, max_seq_len_src, dropout)
        self.wpe_tgt  = PositionalEncoding(d_model, max_seq_len_tgt, dropout)
        self.encoder = nn.ModuleList([Encoder(d_model,max_seq_len_src,h,dropout,hidden_dropout,output_dropout,mlp_ratio) for _ in range(N)])
        self.decoder = nn.ModuleList([Decoder(d_model,h,dropout,hidden_dropout,output_dropout,mlp_ratio,max_seq_len_src,max_seq_len_tgt) for _ in range(N)])
        self.projection   = ProjectionLayer(d_model,tgt_vocabulary_size)
    

        self._init_weights()

    def _init_weights(self):

        for p in self.parameters():

            if p.dim() > 1:

                nn.init.xavier_uniform_(p)

    def encode(self,src:torch.Tensor,src_mask:Optional[torch.Tensor] = None):

        src = self.wpe_src(self.src_embed(src))

        for layer in self.encoder:

            src = layer(src,src_mask)

        return src    
    
    def decode(self,src:torch.Tensor,tgt:torch.Tensor,src_mask:Optional[torch.Tensor]=None,tgt_mask:Optional[torch.Tensor] = None):

        tgt = self.wpe_tgt(self.tgt_embed(tgt))

        for layer in self.decoder:

            tgt = layer(src,tgt,src_mask,tgt_mask)

        return tgt 
    
    def forward(self,src:torch.Tensor,tgt:torch.Tensor,src_mask:Optional[torch.Tensor]=None,tgt_mask:Optional[torch.Tensor]=None):

        src = self.encode(src,src_mask)
        tgt = self.decode(src,tgt,src_mask,tgt_mask)
        tgt = self.projection(tgt)

        return tgt

    @torch.no_grad()
    def inference(self,src:torch.Tensor,bos_token_id:int,eos_token_id:int,max_seq_len:int):
        
        device = src.device
        src_context = self.encode(src,src_mask=None)                                     
        output = torch.tensor([bos_token_id],dtype=torch.long,device=device).reshape(1,1)
 
        for _ in range(max_seq_len):

            last_logit = self.decode(src_context,output,src_mask=None,tgt_mask=None)[:,-1,:]
            next_token = self.projection(last_logit).argmax(dim=-1,keepdim=True)                                             
            output = torch.cat([output,next_token],dim=1)
            print(output)               
 
            if next_token.item() == eos_token_id:

                break
 
        return output


if __name__ == "__main__":
    src = torch.randint(0,1000, (1,128))#.to("cuda")
    model = Transformer(8,1500,2000,128,128,2,0.1,0.1,0.1,4,3)
    model.inference(src,2,3,25)
            


