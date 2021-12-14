# 导入模块用于获取股票数据
import tushare as ts
# 导入数据处理模块
import numpy as np
import pandas as pd
# 导入神经网络模块
from torch.autograd import Variable
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
# 导入可视化模块
import matplotlib as plt

# 设置个人token
ts.set_token('801000ee55cfd2cd9d915ce6c347689b938de98c482b853c0c7cb66d')
pro = ts.pro_api()
# 获得股票代码
stock = pro.query('stock_basic', exchange='', list_status='L', name='贵州茅台',
                  fields='ts_code,symbol,name,area,industry')
print(stock)
# 获取股票日线数据
stock_data = pro.daily(ts_code='600519.SH', start_date='20180701', end_date='20210718')
print(stock_data.columns)
# 数据预处理
stock_data.drop('ts_code', axis=1, inplace=True)  # 删除第二列’股票代码‘
stock_data.drop('pre_close', axis=1, inplace=True)  # 删除列’pre_close‘
stock_data = stock_data.sort_index(ascending=False)
stock_data = stock_data.set_index('trade_date')  # 将列’trade_date‘设置为索引
print(stock_data.columns)

close_max = stock_data['close'].max()  # 收盘价的最大值
close_min = stock_data['close'].min()  # 收盘价的最小值
# print(close_min) 524
# print(close_max) 2601
# print(stock_data)

df = stock_data.apply(lambda x: (x - min(x)) / (max(x) - min(x)))  # min-max标准化
print(df)

# 构造x和y
# 根据前n天的数据，预测未来一天的收盘价(close)。
# 例如：根据1月1日、1月2日、1月3日、1月4日、1月5日的数据（每一天的数据包含8个特征），预测1月6日的收盘价。
sequence = 5
X = []
Y = []
for i in range(df.shape[0] - sequence):  # 741 - 5
    X.append(np.array(df.iloc[i:(i + sequence), :].values, dtype=np.float32))
    Y.append(np.array(df.iloc[(i + sequence), 0], dtype=np.float32))
print(X[0])
print(Y[0])
print(df.shape)  # (741,8)
# 构建batch
total_len = len(Y)
print(total_len)  # 736
batchSize = 64


class Mydataset(Dataset):
    def __init__(self, xx, yy, transform=None):
        self.x = xx
        self.y = yy
        self.tranform = transform

    def __getitem__(self, index):
        x1 = self.x[index]
        y1 = self.y[index]
        if self.tranform is not None:  # transform != None
            return self.tranform(x1), y1
        return x1, y1

    def __len__(self):
        return len(self.x)


print(int(0.99 * total_len))  # 728
trainx, trainy = X[:int(0.99 * total_len)], Y[:int(0.99 * total_len)]
testx, testy = X[int(0.99 * total_len):], Y[int(0.99 * total_len):]
train_loader = DataLoader(dataset=Mydataset(trainx, trainy, transform=transforms.ToTensor()), batch_size=batchSize,
                          shuffle=True)
test_loader = DataLoader(dataset=Mydataset(testx, testy), batch_size=batchSize, shuffle=True)


'''
定义并训练LSTM模型
'''


class lstm(nn.Module):

    def __init__(self, input_size=8, hidden_size=32, num_layers=1, output_size=1, dropout=0, batch_first=True):
        super(lstm, self).__init__()
        # lstm的输入 #batch,seq_len, input_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.batch_first = batch_first
        self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                           batch_first=self.batch_first, dropout=self.dropout)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        out, (hidden, cell) = self.rnn(x)
        # x.shape : batch, seq_len, hidden_size , hn.shape and cn.shape :
        # num_layers * direction_numbers, batch, hidden_size
        # a, b, c = hidden.shape
        # out = self.linear(hidden.reshape(a * b, c))
        out = self.linear(hidden)
        return out


model = lstm(input_size=8, hidden_size=32, num_layers=2, output_size=1,
             dropout=0.1, batch_first=True)
criterion = nn.MSELoss()  # 定义损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Adam梯度下降  学习率=0.0001

epochs = 100
for i in range(epochs):
    total_loss = 0
    for idx, (data, label) in enumerate(train_loader):
        data1 = data.squeeze(1)
        pred = model(Variable(data1))
        pred = pred[1, :, :]
        label = label.unsqueeze(1)
        loss = criterion(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(total_loss)
    if i % 10 == 0:
        # torch.save(model, args.save_file)
        torch.save({'state_dict': model.state_dict()}, 'stock.pkl')
        print('第%d epoch，保存模型' % i)
# 保存模型
torch.save({'state_dict': model.state_dict()}, 'stock.pkl')





# 加载训练好的模型
checkpoint = torch.load('stock.pkl')
model.load_state_dict(checkpoint['state_dict'])
preds = []
labels = []
for idx, (x, label) in enumerate(test_loader):
    x = x.squeeze(1)
    pred = model(x)
    list = pred.data.squeeze(1).tolist()
    preds.extend(list[-1])
    labels.extend(label.tolist())
for i in range(len(preds)):
    print('预测值是%.2f,真实值是%.2f' % (
        np.mat(preds[i]) * (close_max - close_min) + close_min, labels[i] * (close_max - close_min) + close_min))
print(len(preds))  # 8
