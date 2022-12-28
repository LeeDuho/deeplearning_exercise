# Duho Lee

import torch
import numpy as np

#%%
# solution of problem 1

x1 = torch.tensor([1,2,3,4])
print(x1)
y1 = x1.view(2,2)
print(y1)


#%%
# solution of problem 2

x2 = torch.arange(5,101,5)
print(x2)

y2 = x2[3:len(x2):4]
print(y2)
    
#%%
# solution of problem 3

x3 = torch.tensor([1,2,3])
print(x3)
y3_1 = x3.index_select(dim=-1,index = torch.tensor([0,0,0,1,1,1,2,2,2]))
y3_2 = x3.index_select(dim=-1,index = torch.tensor([0,1,2,0,1,2,0,1,2]))
y3 = torch.cat([y3_1,y3_2], dim = 0 )
print(y3)

#%%
# solution of problem 4

x4 = torch.tensor([1,0,1,1])
y4 = x4.bool()

print(y4)


#%%
# solution of problem 5

x5 = torch.randn(2,3,4) * 2 + 10
print(x5)

#%%
# solution of problem 6

y6 = x5.view([1,-1])
y6 = y6.squeeze()

print(y6)
print(y6.size())


#%%
# solution of problem 7

y6_max = y6.max()
y6_min = y6.min()
print(y6_max, y6_min)
y7 = (y6 - y6_min) / (y6_max - y6_min)

print(y7)


#%%
# solution of problem 8

x8 = torch.rand(100) * 20 -10
print(x8)

x8_max = x8.max()
x8_min = x8.min()
x8_mean = x8.mean()
x8_std = x8.std()
x8_median = x8.median()
x8_variance = x8_std * x8_std
x8_mode = x8.mode()

print(x8_max, x8_min, x8_mean ,x8_std ,x8_median ,x8_variance ,x8_mode)

#%%
# solution of problem 9

x9_r = torch.rand(64)
x9_i = torch.rand(64)
x9 = torch.complex(x9_r, x9_i)

x9 = x9.view([8,-1])

x9_t = torch.transpose(x9, 0, 1)

x9_result = torch.mm(x9, x9_t)

print(x9)
print(x9_t)
print(x9_result)


#%%
# solution of problem 10


x10 = torch.randint(0, 10, size = (10,10))

print(x10)

for i in range(10):
  for j in range(10):
    if x10[i][j] == 5:
      print(" row:", i," col:", j)

