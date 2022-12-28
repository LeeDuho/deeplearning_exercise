# Duho Lee

from pickletools import optimize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()

X = digits.data
Y = digits.target
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.int64)
X_train, X_test = torch.split(X, 1497)
Y_train, Y_test = torch.split(Y, 1497)

ds_train = TensorDataset(X_train, Y_train)
ds_test = TensorDataset(X_test, Y_test)

loader_train = DataLoader(ds_train, batch_size=64, shuffle=True)
loader_test = DataLoader(ds_test, batch_size=64, shuffle=False)

net = nn.Sequential(
    nn.Linear(64, 32), 
    nn.ReLU(),
    nn.Linear(32, 16), 
    nn.ReLU(),
    nn.Linear(16, 10)
)

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters())
lr = 0.1

losses = []

net.train()
for epoch in range(300):
  optimizer.zero_grad()
  y_pred = net(X)
  loss = loss_fn(y_pred, Y)
  loss.backward()
  optimizer.step()

  losses.append(loss.item())

plt.plot(losses)
plt.show()

net.eval()

correct = 0

with torch.no_grad():
  for data, targets in loader_test:
    outputs = net(data)

    _, predicted = torch.max(outputs.data, 1) 
    correct += predicted.eq(targets.data.view_as(predicted)).sum()

data_num = len(loader_test.dataset)
print('\n테스트 데이터에서 예측 정확도: {}/{} ({:.0f}%)\n'.format(correct, data_num, 100. * correct / data_num))

