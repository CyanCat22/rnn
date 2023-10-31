import torch
from d2l import torch as d2l
from torch import nn
"""
编码器 特征提取 将文本表示成向量 处理输出 将长度可变的序列作为输入，转换为具有固定形状的编码状态
解码器 解码成输出 向量表示成输出 生成输出 将具有固定形状的编码状态映射为长度可变的序列
"""
class EncoderDeconder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDeconder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)

 