# Duho Lee

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch import nn, optim
import tqdm

import random
import numpy as np

import torchvision
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

device= torch.device("cuda:0" if torch.cuda.is_available() else"cpu")


cstr= "deep learning programming "
chars = ['d', 'a', ' ', 'm', 'e', 'n', 'p', 'r', 'i', 'l', 'g', 'o']
#chars = list(set(cstr))
print(chars)

char2num = {c: i for i, c in enumerate(chars)}
num2char = {i: c for i, c in enumerate(chars)}

tb = []
for c in cstr:
  tb.append(char2num[c])

# input data, batch
onehotencoding= F.one_hot(torch.tensor([char2num[c] for c in cstr]))
x_batch = torch.tensor(onehotencoding)
x_batch = torch.tensor(x_batch, dtype = torch.float32)

# target data, batch
t_batch = torch.tensor(tb)

x = x_batch[:-1]
t = t_batch[1:]


class CharRNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super().__init__()
    self.hidden_size = hidden_size
    self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
    self.fc = nn.Linear(hidden_size, output_size)
    self.softmax = nn.LogSoftmax(dim=1)

  def forward(self, x, h=None):
    if h is None:
      h = torch.zeros(1, self.hidden_size)
    out, hn = self.rnn(x, h)
    y = self.fc(out)
    #return y
    return self.softmax(y)
  
input_size, hidden_size = 12, 10
rnn = CharRNN(input_size, hidden_size, input_size)
optimizer = optim.Adam(rnn.parameters(), lr=0.01)
lossfn = nn.NLLLoss()
losses = []
rnn.train()

for epoch in tqdm.tqdm(range(1000)):
  y = rnn(x)
  L = lossfn(y.view(-1, len(chars)), t.view(-1))

  optimizer.zero_grad()
  L.backward()
  optimizer.step()
  losses.append(L.item())

plt.plot(losses)
plt.grid()
plt.show()

rnn.eval()
y = rnn(x)
print("   input: ", end="")
for c in cstr[:-1]:
  print(c, end="|")
print("")
print("predicted: ", end="")  

#print predicted vector, y
for i in range(len(y)):
  _, idx = torch.max(y[i], dim=0)
  print(num2char[idx.item()], end="|")

