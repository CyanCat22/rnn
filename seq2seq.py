import torch
from d2l import torch as d2l
from torch import nn 
import collections
import math

"""
编码器是一个RNN(可以使双向),读取输入句子
解码器用另一个RNN来输出
编码器最后时间步的隐状态用作解码器的初始隐状态
训练师解码器使用目标句子作为输入
BLEU评估衡量
"""
class Seq2SeqEncoder(d2l.Encoder):
    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层获得每个词元的特征向量，
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
         # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding
        # 在循环神经网络模型中，第一个轴对应于时间步
        # 变换X的形状，将序列长度seq_length放在了第二个维度，时间步交换
        X = X.permute(1, 0, 2)
        output, state = self.rnn(X)
        # output的形状:(num_steps,batch_size,num_hiddens)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)




