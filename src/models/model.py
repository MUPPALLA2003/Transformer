import torch
import torch.nn as nn
from InputEmbed import InputEmbeddings
from PositionalEncoding import PositionalEncoding
from LayerNorm import LayerNormalization
from Encoder import Encoder
from Decoder import Decoder
from Projection import ProjectionLayer
class Transformer(nn.Module):

    def __init__(self, d_model: int, src_vocabulary_size: int, tgt_vocabulary_size: int,max_seq_len_src:int,max_seq_len_tgt:int,h: int, dropout: float, causal_mask: bool,hidden_dropout: float, output_dropout: float,mlp_ratio: int, N: int, output_vocab_size: int):

        super().__init__()

        self.src_embed = InputEmbeddings(d_model, src_vocabulary_size)
        self.tgt_embed = InputEmbeddings(d_model, tgt_vocabulary_size)
        self.wpe_src      = PositionalEncoding(d_model, max_seq_len_src, dropout)
        self.wpe_tgt       = PositionalEncoding(d_model, max_seq_len_tgt, dropout)
        self.encoder = nn.ModuleList([Encoder(d_model,max_seq_len_src,h,dropout,causal_mask,hidden_dropout,output_dropout,mlp_ratio) for _ in range(N)])
        self.decoder = nn.ModuleList([Decoder(d_model,h,dropout,hidden_dropout,output_dropout,mlp_ratio,max_seq_len_src,max_seq_len_tgt) for _ in range(N)])
        self.encoder_norm = LayerNormalization(d_model)
        self.decoder_norm = LayerNormalization(d_model)
        self.projection   = ProjectionLayer(d_model,output_vocab_size)

        self._init_weights()

    def _init_weights(self):

        for p in self.parameters():

            if p.dim() > 1:

                nn.init.xavier_uniform_(p)



        


