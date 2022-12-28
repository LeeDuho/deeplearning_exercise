# Duho Lee

import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np

import torchvision
from torch.utils.data import DataLoader

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torchvision import datasets

import plotly.express as px
import pandas as pd

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

dataset = torchvision.datasets.FashionMNIST(
  root='data',
  download=True,
  train=True,
  transform=torchvision.transforms.ToTensor()
)

class_names = ['T-shirt/top', 'Trousera', 'Pullover', 'Dress', 'Coat',
  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

images = dataset.data.view(-1, 28*28).numpy()
ssc = StandardScaler()
images = ssc.fit_transform(images)
labels_num = dataset.targets.numpy()
labels = [class_names[i] for i in labels_num]

pca_result = PCA(n_components=3).fit_transform(images)

loader= DataLoader(dataset, batch_size=300, shuffle=True, drop_last=True)

#Autoencoder 구현
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 28*28),
            #last layer is sigmiod
            nn.Sigmoid())
            
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Autoencoder()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
loss_fn = nn.MSELoss()

num_epochs = 2
for epoch in range(num_epochs):
    for data in loader:
        img, _ = data  
        img = img.view(img.size(0), -1)
        img = img
        # ===================forward=====================
        output = model(img)
        loss = loss_fn(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, num_epochs, loss.item()))

#Visualize extracted features on 3-d space
model.eval()
with torch.no_grad():
    images = dataset.data.view(-1, 28*28)
    images = images.float()
    encoded_data = model.encoder(images)
    encoded_data = encoded_data.numpy()

x = pd.DataFrame(encoded_data)
x['labels'] = labels
x.head()
fig = px.scatter_3d(x, x=0, y=1, z=2, color = 'labels')
fig.update_traces(marker_size=1)
fig.show()

#Plot input images and decoded output images (n=5)
n = 5
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(images[i].view(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(output[i].view(28, 28).detach().numpy()) 
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()




# ML분야에서 이야기하는 manifold란 무엇인지 조사

# Manifold란 고차원의 데이터를 저차원으로 표현할 수 있는 공간을 의미한다.
# 고차원으로 갈수록 데이터들은 희박하게 분포하게 된다. 
# 이럴 경우 비슷한 데이터들의 특성을 잘 표현할 수 없다. 
# 따라서 차원축소로 차원의 저주 문제를 해결하고, 학습 속도와 모델 성능을 향상시키기 위해 
# Manifold Learning이 사용된다. 
# Manifold Learning은 고차원 데이터가 있을 때 고차원 데이터를 데이터 공간에 뿌리면 
# 샘플들을 잘 아우르는 subspace가 있을 것이라 가정에서 학습을 진행하는 방법이다. 
# Manifold Learning은 차원축소를 위해 사용하며 이를 통해 고차원 데이터를 저차원에서도 잘 표현하는 공간인 manifold를 찾아 차원을 축소시킨다.
