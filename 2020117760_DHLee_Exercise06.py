# Duho Lee

import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np

import torchvision
from torch.utils.data import DataLoader

# CPU
random_seed= 7
torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)
# GPU
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


batch_size= 10
num_output_nodes= 3
reduction= "sum" # or "mean", depending

y= torch.randn((batch_size, num_output_nodes)) # output from 3 class
t= torch.randint(0, num_output_nodes, (batch_size, ))

lossfn_ce= nn.CrossEntropyLoss(reduction=reduction)
lsm= nn.LogSoftmax(dim=1)
nll= nn.NLLLoss(reduction=reduction)

print("loss by NLL + LogSoftmax :", nll(lsm(y), t))
print("loss by CrossEntropyLoss :", lossfn_ce(y, t))
print(y, t)

device= torch.device("cuda:0" if torch.cuda.is_available() else"cpu")

dataset = torchvision.datasets.FashionMNIST(
  root='./data',
  download=True,
  train=True,
  transform=torchvision.transforms.ToTensor()
)
loader= DataLoader(dataset, batch_size=300, shuffle=True, drop_last=True)

class ConvNet(torch.nn.Module):
  def __init__(self):
    super(ConvNet,self).__init__()
    self.conv_layer_1= torch.nn.Sequential( 
      torch.nn.Conv2d(1, 64, kernel_size = (3, 3)),
      torch.nn.BatchNorm2d(64),
      torch.nn.ReLU(),
      torch.nn.MaxPool2d(kernel_size = 2)
    )
    self.conv_layer_2= torch.nn.Sequential(
      torch.nn.Conv2d(64, 32, kernel_size = (3, 3)),
      torch.nn.BatchNorm2d(32),
      torch.nn.ReLU(),
      torch.nn.MaxPool2d(kernel_size = 2)
    )
    self.fcn= torch.nn.Sequential(
      torch.nn.Linear(800, 128),
      torch.nn.ReLU(),
      torch.nn.Dropout(p=0.3),
      torch.nn.Linear(128, 64),
      torch.nn.ReLU(),
      torch.nn.Dropout(p=0.3),
      torch.nn.Linear(64, 10),
      torch.nn.LogSoftmax()
    )

  def forward(self, x):
    out = self.conv_layer_1(x)
    out = self.conv_layer_2(out)

    out = out.view(out.size(0), -1)   # Flatten 
    out = self.fcn(out)
    return out

net= ConvNet().to(device=device)

loss_fn = nn.CrossEntropyLoss(reduction=reduction).to(device)
optimizer = optim.Adam(net.parameters(), lr=0.002)

net.train()

for epoch in range(5):
  for X, t in loader: 

    X = X.to(device)
    t = t.to(device)

    optimizer.zero_grad()
    t_pred = net(X)
    loss = loss_fn(t_pred, t)
    loss.backward()
    optimizer.step()


net.eval()

testdata = torchvision.datasets.FashionMNIST(
  root='./data',
  train=False,
  transform=torchvision.transforms.ToTensor()
)
tdldr = DataLoader(testdata, batch_size=600, shuffle=False, drop_last=True)

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

print('Accuracy:', (result_accuracy/cnt).item())



# Homework #4
# 1) Complete your code
#   •Report your accuracy after 3 epoch of iteration
#    
#   accuracy = 0.87677 <- 3 epoch (batch normalization, dropout layer 사용하지 않았을때)
# 
# 2) Improve overall accuracy (>90%)
#   •max. epoch allowed = 5
#   •use batch normalization and/or dropout layer
#   •Report overall and class accuracy
#
#   accuracy = 0.90041 <- 5 epoch, batch normalization, dropout layer 사용시 
#