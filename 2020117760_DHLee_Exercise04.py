# Duho Lee

#%%
# Logical AND/OR/XOR training 
# 1. Design and train a network to solve AND, OR, XOR problem all in one.
# (a) Design a network and explain how did you come up with such an idea.
#   -똑같이 sigmoid 함수를 사용하였고 입력값이 3개로 변하여 입력값의 개수를 조정했습니다
# (b) Write function that generate input data (you can use your previous homework code as well).
#   -torch.cat 함수를 하용하여 입력 데이터를 하나로 만들고, op를 각각 0,1,2로 표현해
#     input으로 주었습니다
# (c) Choose appropriate activation function, loss function and optimizer.
# (d) If you got trouble in training, explain what was the problem and how to solve it (You can write an essay here).

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import LogicDataset
import matplotlib.pyplot as plt

# AND/OR
data1 = LogicDataset(300, op='and')
data2 = LogicDataset(300, op='or')
data3 = LogicDataset(300, op='xor')

#data.add_noise(0.001)
# x = data.X, data.Y

data1.add_noise(0.001)
data2.add_noise(0.001)
data3.add_noise(0.001)

x1 = torch.tensor(data1.X, dtype=torch.float32)
xx1= torch.cat((x1,torch.zeros(1200, 1, dtype=torch.float32)),1)
x2 = torch.tensor(data2.X, dtype=torch.float32)
xx2= torch.cat((x2,torch.ones(1200, 1, dtype=torch.float32)),1)
x3 = torch.tensor(data3.X, dtype=torch.float32)
xx3= torch.cat((x3,torch.ones(1200, 1, dtype=torch.float32)+1.0),1)

x = torch.cat((xx1,xx2,xx3),0)

print(x)

y1 = torch.tensor(data1.Y, dtype=torch.float32)
y2 = torch.tensor(data2.Y, dtype=torch.float32)
y3 = torch.tensor(data3.Y, dtype=torch.float32)

y = torch.cat((y1,y2,y3),0)

print(y)

# AND/OR
net = nn.Sequential(
    nn.Linear(3, 1),
    nn.Sigmoid()
)
# XOR
net = nn.Sequential(
    nn.Linear(3, 2),
    nn.Sigmoid(),
    nn.Linear(2, 1),
    nn.Sigmoid()
)

lossfn = nn.MSELoss()
lr = 0.5
# optimizer = optim.SGD(net.parameters(), lr)
optimizer = optim.Adam(net.parameters())

losses = []
for epoch in range(5000):

    optimizer.zero_grad()

    y_pred = net(x)

    L = lossfn((y_pred), y)
    L.backward()

    optimizer.step()
    
    losses.append(L.item())

plt.plot(losses)
plt.grid()
plt.show()

plt.plot(y)
plt.plot(y_pred.detach())
plt.show()



###################################################################
#%%

# 2. Design and train a network to classify the spiral data.
# (a) Design a network and explain how did you come up with such an idea.
#     -5layer의 network를 만들었고 sigmoid 함수를 사용하여 epoch를 10000으로 설정했습니다
# (b) Choose appropriate activation function, loss function and optimizer.
# (c) If you got trouble in training, explain what was the problem and how to solve it (You can write an essay here).


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from datasets import SpiralDataset

data = SpiralDataset(300)
data.generate()

data.plot()

data_x, data_t = data.get_data()

x = torch.tensor(data_x)
t = torch.tensor(data_t)
x= x.squeeze()
t= t.squeeze()

print(x)
print(t)

net = nn.Sequential(
    nn.Linear(2, 3),
    nn.Sigmoid(),

    nn.Linear(3, 4),
    nn.Sigmoid(),
    nn.Linear(4, 5),
    nn.Sigmoid(),
    nn.Linear(5, 4),
    nn.Sigmoid(),
    
    nn.Linear(4, 3),
    nn.Sigmoid()
    
)

lossfn = nn.MSELoss()
lr = 0.3
# optimizer = optim.SGD(net.parameters(), lr)
optimizer = optim.Adam(net.parameters())

losses = []
for epoch in range(10000):

    optimizer.zero_grad()

    t_pred = net(x)

    L = lossfn((t_pred), t)
    L.backward()

    optimizer.step()
    
    losses.append(L.item())

plt.plot(losses)
plt.grid()
plt.show()

plt.plot(t)
plt.plot(t_pred.detach())
plt.show()

