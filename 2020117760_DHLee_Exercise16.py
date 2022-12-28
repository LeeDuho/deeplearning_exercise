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

# implementing Deep Convolutional GAN(DCGAN)

transform= transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize([0.5], [0.5]),
])

dataset = torchvision.datasets.MNIST(
  root='./data',
  download=True,
  train=True,
  transform=transform
)
loader= DataLoader(dataset, batch_size=100, shuffle=True, drop_last=True)

class Generator(nn.Module):
  def __init__(self, lat_dim=64):
    super().__init__()
    self.input_size= 7
    self.stage1 = nn.Linear(lat_dim, 128 * self.input_size ** 2)
    self.stage2 = nn.Sequential(
      nn.ConvTranspose2d(128, 64, 4, 2, 1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.ConvTranspose2d(64, 1, 4, 2, 1),
      nn.Tanh()
    )
  def forward(self, x):
    x= self.stage1(x)
    x= x.view(x.shape[0], 128, self.input_size, self.input_size)
    out= self.stage2(x)
    return out

class Discriminator(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    
    def disc_module(in_ch, out_ch):
      mod = [
        nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
      ]
      return mod

    self.disc_model= nn.Sequential(
      *disc_module(1, 512),
      *disc_module(512, 256),
      *disc_module(256, 128),
      nn.AvgPool2d(4),
    )

    self.fc= nn.Sequential(
      nn.Linear(128, 1),
      nn.Sigmoid(),
    )

  def forward(self, x):
    x= self.disc_model(x)
    x= x.view(x.shape[0], -1)
    out= self.fc(x)
    return out

G = Generator().to(device)
D = Discriminator().to(device)

loss_fn = nn.BCELoss()

opt_G = optim.Adam(G.parameters(), lr=0.01, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=0.01, betas=(0.5, 0.999))

lat_dim = 64
num_epochs = 10


for epoch in range(num_epochs):
  for idx, (images, _) in tqdm(enumerate(loader)):
    
    T = torch.ones(images.shape[0], 1, requires_grad=True).to(device)

    z = torch.randn((images.shape[0], lat_dim)).to(device)
    gen_images = G(z)
    loss_G = loss_fn(D(gen_images), T)

    opt_G.zero_grad()
    loss_G.backward()
    opt_G.step()

    #training the discriminator
    F = torch.zeros(images.shape[0], 1, requires_grad=True).to(device)
    
    real_images = images.type(torch.FloatTensor).to(device)

    real_image_loss = loss_fn(D(real_images), T)
    fake_image_loss = loss_fn(D(gen_images.detach()), F)

    loss_D = (real_image_loss + fake_image_loss) / 2

    opt_D.zero_grad()
    loss_D.backward()
    opt_D.step()

    #Save 100 output images (the last batch of each epoch) for each epoch
    if idx == len(loader) - 1:
      for i in range(100):
        #save image
        torchvision.utils.save_image(gen_images[i], f'./output/{epoch}_{i}.png')


