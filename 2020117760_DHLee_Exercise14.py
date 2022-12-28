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

num_epochs = 100
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

#모든 이미지들의 Encoder 결과를 저장
encoded_imgs = []
for data in loader:
    img, _ = data  
    img = img.view(img.size(0), -1)
    img = img
    encoded_imgs.append(model.encoder(img).detach().numpy())

encoded_imgs = np.concatenate(encoded_imgs, axis=0)

#가장 유사한 이미지 top5를 검색(encoding된 3차원 벡터의 거리를 이용)
def find_similar_images(img, top=5):
    model.eval()
    with torch.no_grad():
        img = img.view(-1, 28*28)
        img = img.float()
        output = model.encoder(img)
        output = output.numpy() #encoding된 3차원 벡터
        img = img.numpy() #img는 원본 이미지, output은 encoding된 3차원 벡터
        #output과 다른 모든 encoding 된 이미지들과의의 거리를 계산
        distances = np.linalg.norm(output - encoded_imgs, axis=1)
        indices = np.argsort(distances) # 오름차순 정렬

        indices = indices[:top] #가장 유사한 top5 이미지의 인덱스, 자기 자신은 제외

        #indices의 이미지들을 출력, 0번째는 자기 자신을 출력
        fig, axes = plt.subplots(1, top + 1, figsize=(20, 3))  # 1행 5열의 그래프, sigsize는 그래프의 크기
        axes[0].imshow(img.reshape(28, 28), cmap='gray') #원본 이미지 출력
        axes[0].set_title('Original Image')
        for i in range(top):
            axes[i+1].imshow(images[indices[i]].reshape(28, 28), cmap='gray')
            axes[i+1].set_title('distance: {:.2f}'.format(distances[indices[i]]))
       
        plt.show()


#검색하고자 하는 이미지를 선택, 출력
img = dataset.data[0]
find_similar_images(img)

