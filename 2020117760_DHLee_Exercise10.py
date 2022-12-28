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

#####-----------CBOW-----------#####
class CBOW(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(CBOW, self).__init__()
    self.embeddings = nn.Embedding(input_size, hidden_size)
    self.linear1 = nn.Linear(hidden_size, input_size)

  def forward(self, inputs):
    embeds = torch.mean(self.embeddings(inputs), dim=0).view((1, -1))
    out = self.linear1(embeds)
    log_probs = nn.functional.log_softmax(out, dim=1)
    return log_probs

cbow = CBOW(len(word_to_ix), 10)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(cbow.parameters(), lr=0.0005)
losses = []

for epoch in range(100):
  total_loss = 0
  for i in range(2, len(words)-2):
    context = [words[i-2], words[i-1], words[i+1], words[i+2]]
    target = words[i]
    context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
    cbow.zero_grad()
    log_probs = cbow(context_idxs)
    loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
  losses.append(total_loss)

plt.plot(losses)
plt.show()

#plot word vectors in 2d space
word_vectors = cbow.embeddings.weight.data.numpy()
word_vectors = word_vectors.T
plt.scatter(word_vectors[0], word_vectors[1])
for i, word in enumerate(word_to_ix):
  plt.annotate(word, (word_vectors[0][i], word_vectors[1][i]))
plt.show()

#####-----------Skipgram-----------#####
class Skipgram(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(Skipgram, self).__init__()
    self.embeddings = nn.Embedding(input_size, hidden_size)
    self.linear1 = nn.Linear(hidden_size, input_size)

  def forward(self, inputs):
    embeds = self.embeddings(inputs).view((1, -1))
    out = self.linear1(embeds)
    log_probs = nn.functional.log_softmax(out, dim=1)
    return log_probs

skipgram = Skipgram(len(word_to_ix), 10)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(skipgram.parameters(), lr=0.0005)
losses = []

for epoch in range(100):
  total_loss = 0
  for i in range(2, len(words)-2):
    
    target = words[i]

    skipgram.zero_grad()
    log_probs = skipgram(torch.tensor([word_to_ix[target]], dtype=torch.long))

    context = [words[i-2], words[i-1], words[i+1], words[i+2]]

    loss = 0
    for c in context:
      loss += loss_function(log_probs, torch.tensor([word_to_ix[c]], dtype=torch.long))
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
  losses.append(total_loss)
  
plt.plot(losses)
plt.show()

#plot word vectors in 2d space
word_vectors = skipgram.embeddings.weight.data.numpy()
word_vectors = word_vectors.T
plt.scatter(word_vectors[0], word_vectors[1])
for i, word in enumerate(word_to_ix):
  plt.annotate(word, (word_vectors[0][i], word_vectors[1][i]))
plt.show()
