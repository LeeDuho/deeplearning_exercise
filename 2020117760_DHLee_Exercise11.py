# Duho Lee

# Implement CBoW and Skipgram
# Dataset peter.txt
# load and preprocess text data
# input : nn.Embedding layer with arbitrary window size
# Choose appropriate hidden layer size so that loss -> ~0
# plot word vectors in 2d space (just use the first two dimension)

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

#Preprocessing and cleaning
#tokenize
with open("peter.txt", "r", encoding="utf8") as f:
  txt = f.read()

#stopwords 제거
stop_words = set(stopwords.words('english'))  #stopwords는 불용어를 제거하는 것, 불용어는 자주 쓰이지만 의미가 없는 단어들을 의미한다.
word_tokens = word_tokenize(txt)  
words = []
for word in word_tokens:  #각 단어들을 돌면서
  if word not in stop_words:  #불용어가 아니면
    words.append(word) #result에 추가한다.

#make word_to_ix
word_to_ix = {}
for word in words:
  if word not in word_to_ix:
    word_to_ix[word] = len(word_to_ix)  #word_to_ix에 word를 key로, index를 value로 저장한다.

#make dataset
window_size = 2
idx_pairs = []
label = []
for i in range(window_size, len(words) - window_size):
  for j in range(i - window_size, i + window_size + 1):
    if j == i:
      continue
    idx_pairs.append([word_to_ix[words[i]], word_to_ix[words[j]]])
    label.append(1)
    for k in range(5):
      neg = random.randint(0, len(words) - 1)
      while neg >= i - window_size and neg <= i + window_size:
        neg = random.randint(0, len(words) - 1)
      idx_pairs.append([word_to_ix[words[i]], word_to_ix[words[neg]]])
      label.append(0)

#make dataset
class Dataset(torch.utils.data.Dataset):
  def __init__(self, idx_pairs, label):
    self.idx_pairs = idx_pairs
    self.label = label

  def __getitem__(self, index):
    return self.idx_pairs[index], self.label[index]

  def __len__(self):
    return len(self.idx_pairs)
    
dataset = Dataset(idx_pairs, label)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

#make model
#input: inpu1, input2 (word index), output: label (0 or 1)
class SkipGram(nn.Module):
  def __init__(self, vocab_size, embedding_dim):
    super(SkipGram, self).__init__()
    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim
    self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
    self.linear = nn.Linear(self.embedding_dim, self.vocab_size)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, input1, input2):
    embed1 = self.embedding(input1)
    embed2 = self.embedding(input2)
    out = self.linear(embed1 * embed2)
    out = self.softmax(out)
    return out
    
#make loss function
loss_function = nn.CrossEntropyLoss()
#make optimizer
model = SkipGram(len(word_to_ix), 10)
optimizer = optim.SGD(model.parameters(), lr=0.001)

#train
losses = []
for epoch in range(100):
  total_loss = 0
  for data, target in dataloader:
    model.zero_grad()
    log_probs = model(data[0], data[1])
    loss = loss_function(log_probs, target)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
  losses.append(total_loss)
  print("epoch: ", epoch, "loss: ", total_loss)


#plot
plt.plot(losses)
plt.show()

#plot word vectors in 2d space
word_vectors = model.embedding.weight.data.numpy()
word_vectors = word_vectors.T
plt.scatter(word_vectors[0], word_vectors[1])
for i, word in enumerate(word_to_ix):
  plt.annotate(word, (word_vectors[0][i], word_vectors[1][i]))
plt.show()
