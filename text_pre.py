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
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    
tokens = tokenize(lines)
for i in range(20):
    print(tokens[i])

