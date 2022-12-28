# Duho Lee


import torch
import matplotlib.pyplot as plt

a, b = 2, 3
x = torch.randn(200, 1)
y = a * x + b + torch.randn(200, 1) * 0.3

plt.plot(x, y, '.')
plt.grid()
plt.show()

aa = torch.randn(1, requires_grad=True)
bb = torch.randn(1, requires_grad=True)
lr = 0.1

losses = []

for epoch in range(100):

    aa.grad = None
    bb.grad = None

    y_pred = aa * x + bb

    L = torch.mean((y_pred - y) ** 2)
    L.backward()

    aa.data -= lr * aa.grad.data
    bb.data -= lr * bb.grad.data
    
    losses.append(L.item())

plt.plot(losses)
plt.grid()
plt.show()

print(aa, bb)


#%%
# solution of problem 2

import torch
import matplotlib.pyplot as plt

x = torch.randn(200, 2)
W = torch.tensor([[2.0],[2.0]])
b = 3
y = torch.mm(x, W) + b + torch.randn(200, 1) * 0.3


plt.plot(x, y, '.')
plt.grid()
plt.show()

# initialize coefficients
WW = torch.randn(2,1, requires_grad=True)
bb = torch.randn(1, requires_grad=True)
lr = 0.1

losses = []

for epoch in range(100):

    WW.grad = None
    bb.grad = None

    # feed forward
    y_pred = torch.mm(x, WW) + bb

    # loss
    L = torch.mean((y_pred - y) ** 2)
    L.backward()

    WW.data -= lr * WW.grad.data
    bb.data -= lr * bb.grad.data
    
    losses.append(L.item())

plt.plot(losses)
plt.grid()
plt.show()

print(WW, bb)

