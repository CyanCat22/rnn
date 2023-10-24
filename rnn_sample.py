import torch
from torch import nn 
from torch.nn import functional as F
from d2l import torch as d2l
"""
rnn参数共享
h=a(wh+wh+b)
存储时序信息
"""
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
num_hiddens = 256
rnn_layer = nn.RNN(len(vocab), num_hiddens)

state = torch.zeros((1, batch_size, num_hiddens))
print(state.shape)

X = torch.rand(size=(num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X , state)
print(Y.shape, state_new.shape)

class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 是否双向
        if not self.rnn.bidirectional:
            self.num_directions = 1
            # 构造自己的输出层
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        


