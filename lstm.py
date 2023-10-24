import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import seaborn as sns
# 图形可视库
flight_data = sns.load_dataset("flights")
# print(flight_data.head())
# 打印前五行数据

print(flight_data.columns)

# 预处理数据将乘客列类型改为浮点数
all_data = flight_data['passengers'].values.astype(float)





