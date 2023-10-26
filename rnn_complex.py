from typing import Any
import torch
from torch import nn 
import math
from torch.nn import functional as F
from d2l import torch as d2l
import tqdm
batch_size, num_steps = 32, 35
X = torch.arange(10).reshape(2, 5)
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
print(len(vocab))
# 独热编码
# F.one_hot(torch.tensor([0, 2]), len(vocab))

# 小批量数据形状（批量大小，时间步数）
# 将词输出为向量
X = torch.arange(10).reshape((2, 5))
print(F.one_hot(X.T, 28).shape) # 转置将时间维度放到前面

# 初始化模型参数
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size
    def normal(shape):
        # 辅助函数，生成一个均值为0方差为1的Tensor
        return torch.randn(size=shape, device=device)*0.01

    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

# 返回隐藏状态，函数返回一个用0补充的张量
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),)

# 在一个时间步内计算隐藏状态和输出
def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        # 更新当前时刻的隐藏状态
        H = torch.tanh(torch.mm(X, W_xh) 
                       + torch.mm(H, W_hh) 
                       + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
        # 列数：vocab_size, 行数：批量大小*时间长度，输出更新后的隐藏状态
    return torch.cat(outputs, dim=0), (H,)

class RNNModel:
    """包装"""
    # 初始化
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn
    def __call__(self, X, state) :
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)
    
    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


num_hiddens = 512

net = RNNModel(len(vocab), num_hiddens, d2l.try_gpu(),
               get_params, init_rnn_state, rnn)
state = net.begin_state(X.shape[0], d2l.try_gpu())
Y, new_state = net(X.to(d2l.try_gpu()), state)
print(Y.shape, len(new_state), new_state[0].shape)

# 预测prefix之后的一个字符 预热
def predict_train(prefix, num_preds, net, vocab, device):
    state = net.begin_state(batch_size=1, device=device)     
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1,1))
    for  y in prefix[1:]: # 预热期
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
        
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
predict_train('time traveller', 10, net, vocab, d2l.try_gpu())

def grad_clipping(net, theta):
    # 梯度裁剪,预防梯度变大
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta/norm

def train_epoch(net, train_iter, loss, 
                updater, device, use_random_iter):
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2) # 训练损失之和，词元数量
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state对于nn.GRU是个张量
                state.detach_()
            else:
                # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)
        metric.add(l*y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


def train(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_train(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in tqdm.tqdm(range(num_epochs)):
        ppl, speed = train_epoch(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))

num_epochs, lr = 500, 1
train(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())
