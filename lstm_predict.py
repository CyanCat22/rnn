import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm

# 图形可视库
flight_data = sns.load_dataset("flights")
# print(flight_data.head())
# 打印前五行数据

# print(flight_data.columns)

# 预处理数据将乘客列类型改为浮点数
all_data = flight_data['passengers'].values.astype(float)

test_data_size = 12
train_data_size = 120

train_data = all_data[:-test_data_size]
test_data = all_data[-test_data_size:]
print(len(train_data)) # 132
print(len(test_data)) # 12

# 数据归一化(-1,1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(train_data.reshape(-1,1))
# 转化为一维张量,view(-1) -1自动计算该维度大小
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)

# 设置训练输入序列滑动窗口长度为12
train_window = 12
# 创建输入输出序列
def creat_inout_sequences(input_data, train_window):
# 接受原始输入数据，并返回一个元组列表
    inout_seq = [] 
    L = len(input_data)
    for i in range(L-train_window):
        train_seq = input_data[i:i+train_window]
        train_label = input_data[i+train_window:i+train_window+1]
        inout_seq.append((train_seq, train_label))
    return inout_seq
    # 每个元组包含一个输入序列和一个对应的标签

train_inout_seq = creat_inout_sequences(train_data_normalized, train_window)


# 继承自PyTorch库的nn.Module类
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size) 

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


model = LSTM()
model.add_module('linear', nn.Linear(100,1))
loss_function = nn.MSELoss() # 交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001) # Adam随机优化方法
print(model)

"""
LSTM(
  (lstm): LSTM(1, 100)
  (linear): Linear(in_features=100, out_features=1, bias=True)
)
"""

epochs = 150
for i in tqdm.tqdm(range(epochs)):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        # 将优化器中的所有梯度清零,在每次进行参数更新之前，调用这个方法来清除之前的梯度信息，避免梯度累积
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size), torch.zeros(1, 1, model.hidden_layer_size))
        # 设置为两个全0张量
        y_pred = model(seq)
        
        single_loss = loss_function(y_pred, labels) # 计算预测值与真实值的损失
        single_loss.backward()
        optimizer.step() # 根据计算出的梯度来更新模型的参数。这会将模型的参数沿着梯度的方向进行更新，以最小化损失函数。
        
    if i % 25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

        

