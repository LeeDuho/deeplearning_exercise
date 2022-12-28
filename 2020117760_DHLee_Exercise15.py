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

#Generative model
#Variational Autoencoder (VAE)

dataset = torchvision.datasets.MNIST(
  root='./data',
  download=True,
  train=True,
  transform=torchvision.transforms.ToTensor()
)
loader= DataLoader(dataset, batch_size=100, shuffle=True, drop_last=True)

class Encoder(nn.Module):
  def __init__(self, input_size, hidden_size, latent_dim):  #input_size : 784, hidden_size : 400, latent_dim : 20
    super().__init__()
    self.fc= nn.Sequential(
      nn.Linear(input_size, hidden_size),
      nn.ReLU(),
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(),
    )
    self.fc1= nn.Linear(hidden_size, latent_dim)
    self.fc2= nn.Linear(hidden_size, latent_dim)
  
  def forward(self, x):
    x= self.fc(x)
    return self.fc1(x), self.fc2(x)

class Decoder(nn.Module):
  def __init__(self, latent_dim, hidden_size, output_size):
    super().__init__()
    self.layer = nn.Sequential(
      nn.Linear(latent_dim, hidden_size),
      nn.ReLU(),
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(),
      nn.Linear(hidden_size, output_size),
      nn.Sigmoid()
    )
  
  def forward(self, x):
    x_hat = self.layer(x)
    return x_hat
    
class VAE(nn.Module):
  def __init__(self, encoder, decoder):
    super().__init__()
    self.encoder= encoder
    self.decoder= decoder

  def forward(self, x):
    mu, logvar= self.encoder(x) #mu는 평균, logvar는 분산
    z= self.reparameterize(mu, logvar)  #z는 latent variable
    x_hat = self.decoder(z) #x_hat은 reconstruction, 즉 원래의 x를 재구성한 것
    return x_hat

  def reparameterize(self, mu, logvar): 
    epsilon = torch.randn_like(logvar).to(device) #epsilon은 노이즈
    std = torch.exp(0.5*logvar) #std는 표준편차, logvar의 제곱근
    z = mu + std * epsilon  #z는 latent variable, mu와 std를 이용해 계산
    return z
    
  def loss(self, x, x_hat, mu, logvar):
    recon_loss= f.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD= -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + KLD


#implement VAE, reparameterization method, and loss function
max_epoch = 10
latent_dim = 2
hidden_size = 400 
input_size = 784  # MNIST의 경우 28*28=784
output_size = 784

encoder = Encoder(input_size, hidden_size, latent_dim)
decoder = Decoder(latent_dim, hidden_size, output_size)
vae = VAE(encoder, decoder).to(device)

optimizer = optim.Adam(vae.parameters(), lr=0.001)

for epoch in range(max_epoch):
  for x, _ in tqdm(loader):
    x = x.view(-1, 784).to(device)
    x_hat = vae(x)
    mu, logvar = vae.encoder(x)
    loss = vae.loss(x, x_hat, mu, logvar)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  print('Epoch: {} Loss: {:.4f}'.format(epoch, loss.item()))


# Draw distribution of z in 2d space (the latent space)
#Using MNIST dataset

z_list = []
x_hat_list = []
label_list = []
for x, label in loader:
  x = x.view(-1, 784).to(device)
  mu, logvar = vae.encoder(x)
  z = vae.reparameterize(mu, logvar)
  x_hat = vae.decoder(z)

  z_list.append(z)
  x_hat_list.append(x_hat)
  label_list.append(label)

z = torch.cat(z_list, dim=0)
label = torch.cat(label_list, dim=0)

z = z.cpu().detach().numpy()
label = label.cpu().detach().numpy()

plt.figure()

#Draw distribution of z in 2d space (the latent space)

plt.scatter(z[:, 0], z[:, 1], c=label, s=1, cmap='tab10')
plt.colorbar()
plt.title('z distribution')
plt.show()

#Reconstruct MNIST data by sampling z from the latent space

x_hat = torch.cat(x_hat_list, dim=0)
x_hat = x_hat.cpu().detach().numpy()

plt.figure()
for i in range(10 * 10):
  plt.subplot(10, 10, i + 1)
  plt.imshow(x_hat[i].reshape(28, 28), cmap='gray')
  plt.axis('off')
plt.show()

