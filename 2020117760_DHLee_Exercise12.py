# Duho Lee


import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

import glob
import unicodedata
import string

import sys
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from torch.autograd import Variable
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from tqdm import tqdm

from torch.utils.data import DataLoader

#import ncutils
from ncutils import *
import torch.nn.functional as f

device= torch.device("cuda:0" if torch.cuda.is_available() else"cpu")

#sequence to sequence , using LSTM, encoder-decoder

class Encoder(nn.Module):
  def __init__(self, vocab_size, wv_size, hidden_size): #vocab_size : 단어의 개수, wv_size : 단어 벡터의 차원, hidden_size : hidden layer의 차원
    super().__init__()
    self.embed = nn.Embedding(vocab_size, wv_size)
    self.embed.weight.data /= 100
    self.rnn = nn.LSTM(wv_size, hidden_size, batch_first=True)  #word vector를 받아서, hidden layer를 출력한다.

  def forward(self, x):
    #return the last hidden state of the encoder
    x = self.embed(x)
    x, _ = self.rnn(x)
    return x[:,-1,:]

    
ncc= NumCharCorpus()  #NumCharCorpus class를 불러온다.
ncc.fliplr_x()
vocab_size= ncc.vocab_size
wv_size= 16 #word vector의 차원
hidden_size= 100  #hidden layer의 차원
batch_size= 64  

ds_train= ncc.get_dataset(train=True)
train_loader= DataLoader(ds_train, batch_size=batch_size)
test_loader= DataLoader(ncc.get_dataset(train=False), batch_size=batch_size)

x, t= next(iter(train_loader)) 
encoder= Encoder(vocab_size, wv_size, hidden_size)
h= encoder(x)
print(h.shape)  # assert (batch_size, hidden_size)

class Decoder(nn.Module):
  def __init__(self, vocab_size, wv_size, hidden_size):
    super().__init__()
    self.embed = nn.Embedding(vocab_size, wv_size)
    self.embed.weight.data /= 100
    self.rnn = nn.LSTM(wv_size, hidden_size, batch_first=True)
    self.linear = nn.Linear(hidden_size, vocab_size)

  def forward(self, x, h):
    x = self.embed(x)
    x, _ = self.rnn(x, (h.unsqueeze(0), h.unsqueeze(0)))
    # x, _ = self.rnn(x, h)#h는 encoder의 마지막 hidden layer를 받아온다.For batched 3-D input, hx and cx should also be 3-D but got (1-D, 1-D) tensors
    x = self.linear(x)
    return x


class Seq2Seq(nn.Module):
  def __init__(self, vocab_size, wv_size, hidden_size):
    super().__init__()
    self.encoder= Encoder(vocab_size, wv_size, hidden_size)
    self.decoder= Decoder(vocab_size, wv_size, hidden_size)

  def forward(self, x, t):
    d_x, d_t= t[:, :-1], t[:, 1:]
    h= self.encoder(x)
    y= self.decoder(d_x, h)
    return y, d_t

#train
model= Seq2Seq(vocab_size, wv_size, hidden_size).to(device)
lossfn= nn.CrossEntropyLoss()
optimizer= optim.Adam(model.parameters(), lr=0.01)

loss = []
accuracy = []

def eval_seq2seq(model, test_loader, device):
  model.eval()
  with torch.no_grad():
    acc = 0
    for x, t in test_loader:
      x = x.to(device)
      t = t.to(device)
      y, dts = model(x, t)
      acc += (y.argmax(dim=2) == dts).float().mean()
  return acc/len(test_loader)


model.train()
for epoch in range(30):
  aloss = 0
  for x, t in tqdm(train_loader):
    x = x.to(device)
    t = t.to(device)
    optimizer.zero_grad()
    y, dts = model(x, t)
    L = lossfn(y.reshape(-1, vocab_size), dts.flatten())
    L.backward()
    optimizer.step()
    aloss += L.item()
  loss.append(aloss/x.shape[0])
  accuracy.append(eval_seq2seq(model, test_loader, device))
  print("epoch=", epoch, loss[-1], accuracy[-1])


plt.subplot(121)
plt.plot(loss)
plt.grid()
plt.xlabel("epoch")
plt.title("loss")

plt.subplot(122)
plt.plot(accuracy, '.:')
plt.grid()
plt.xlabel("epoch")
plt.title("accuracy")
plt.ylim(0, 1)

plt.show()
