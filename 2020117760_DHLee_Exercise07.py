# Duho Lee

import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np

import torchvision
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

device= torch.device("cuda:0" if torch.cuda.is_available() else"cpu")


tf = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
])


train_data = torchvision.datasets.ImageFolder('./data/train', transform=tf)
test_data = torchvision.datasets.ImageFolder('./data/validation', transform=tf)
loader = DataLoader(train_data, batch_size=8, shuffle=True)
tdldr = DataLoader(test_data, batch_size=8, shuffle=False, drop_last=True)

# increased_dataset = torch.utils.data.ConcatDataset([transformed_dataset, original])

net = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.DEFAULT)
with open("imagenet_classes.txt", "r") as f:
  categories = [s.strip() for s in f.readlines()]
img, t = loader.dataset[135] # a cat image (idx135), a dog image (idx=2014)

####
with torch.no_grad():   #top5
  for X, t in loader: 

    X = X.to(device)
    t = t.to(device)

    prediction = net(X)
result = torch.topk(torch.argmax(prediction, 1),5)
print(result)
####

num = 0
accuracy_now = -1.0
accuracy_before = -1.0
while True:
  net = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.DEFAULT)
  with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

  net.classifier[4] = nn.Linear(4096,512)   #알렉스넷 수정
  net.classifier[6] = nn.Linear(512,2)

  optimizer = optim.SGD(net.parameters(), lr=0.001)
  loss_fn = nn.CrossEntropyLoss().to(device)

  net.train()

  num = num + 1
  for epoch in range(num):
    for X, t in loader: 

      X = X.to(device)
      t = t.to(device)
      print("---------------")
      print("epoch: ", epoch)
      print("num: ", num)
      print(X.shape)
      print(t.shape)
      print(t)
      print("---------------")

      optimizer.zero_grad()
      t_pred = net(X)
      loss = loss_fn(t_pred, t)
      loss.backward()
      optimizer.step()


  net.eval()

  with torch.no_grad():
    result_accuracy = 0.0
    cnt = 0
    for X, t in tdldr: 

      X = X.to(device)
      t = t.to(device)

      prediction = net(X)

      correct_prediction = torch.argmax(prediction, 1) == t
      accuracy = correct_prediction.float().mean()
      result_accuracy += accuracy
      cnt = cnt+1
    accuracy_before = accuracy_now
    accuracy_now = (result_accuracy/cnt).item()
    print('Accuracy:', accuracy_now)
    if(accuracy_now < accuracy_before):
      print('best accuracy:', accuracy_before)
      print('best epoch:', num-1)
      break



# epoch 1 : Accuracy: 0.9615384340286255
# epoch 2 : Accuracy: 0.963942289352417
# epoch 3 : Accuracy: 0.9651442170143127
# epoch 4 : Accuracy: 0.963942289352417
# best accuracy: 0.9651442170143127
# best epoch: 3