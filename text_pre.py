import torch
from torch import nn 
from d2l import torch as d2l
import re
import collections

d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt', '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():
    """读取数据集"""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
print("#  文本总行数:{}".format(len(lines)))
print(lines[0])
print(lines[5])

def tokenize(lines, token='word'):
    """词元化---每个文本序列被拆分为一个词元列表"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    
tokens = tokenize(lines)
for i in range(5):
    print(tokens[i])

class Vocab():
    """词表---将字符串词元映射到从后0开始的数字索引中"""
    """得到的统计结果称之为语料corpus"""
    """未知词元<unk> 填充词元<pad> 序列开始词元<bos> 结束词元<eos>"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率
        



