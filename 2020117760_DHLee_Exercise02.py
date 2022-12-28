# Duho Lee


class Stack:
  def __init__(self):
    self.stack = []

  def isEmpty(self):
    return (len(self.stack) == 0)

  def push(self, item):
    self.stack.append(item)
  
  def peek(self):
    if self.isEmpty():
      printf("stack empty")
    else:
      return self.stack[-1]

  def pop(self):
    if self.isEmpty():
      print("Stack empty")
    else:
      return self.stack.pop(-1)

st = Stack()
st.push(10)
st.push(20) 
st.push(30) 
st.push(40) 
print("Size of stack :", len(st.stack)) 
print('First elem :', st.stack[0]) 
print('The top of the stack :', st.peek()) 
print(st.pop()) 
print(st.pop()) 
print(st.pop()) 
print('Size of stack :', len(st.stack)) 

st2 = Stack()
st2.push(3)
st2.push(2)
st2.push(1)
for d in st2.stack:
  print(d)

#%%
# solution of problem 2

def one_hot_encoder_chr(c):
  alphabet = "abcdefghijklmnopqrstuvwxyz"
  li = []
  key = c.lower()
  for i in alphabet:
    if i == key:
      li.append(1)
    else:
      li.append(0)
  return tuple(li)
    
v1 = one_hot_encoder_chr('w')
print(v1)
v2 = one_hot_encoder_chr('D')
print(v2)


#%%
# solution of problem NumPy & Matplotlib 1

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math


class LogicDataset:
  def __init__(self,N,op):
    self.stack = []
    self.N = N
    self.op = op
    self.M0 = np.zeros((N,1))
    self.M1 = np.ones((N,1))
    self.A = np.concatenate((self.M0,self.M0),axis = 1)
    self.B = np.concatenate((self.M0,self.M1),axis = 1)
    self.C = np.concatenate((self.M1,self.M0),axis = 1)
    self.D = np.concatenate((self.M1,self.M1),axis = 1)
    self.X = np.concatenate((self.A,self.B,self.C,self.D), axis= 0)
    self.Y = np.zeros((N*4,1))

    x1,x2 = np.split(self.X,[1],axis=1)
    if self.op == "and":
      self.Y = x1 * x2
    elif self.op == "or":
      for i in range(N * 4):
        if x1[i][0] == 0 and x2[i][0] == 0:
          self.Y[i][0] = 0
        else:
          self.Y[i][0] = 1
    elif self.op == "xor":
      for i in range(N * 4):
        if x1[i][0] == x2[i][0]:
          self.Y[i][0] = 0
        else:
          self.Y[i][0] = 1
    self.Y = np.concatenate((self.X,self.Y),axis = 1)

  def get_pri_logic_inputs(self,input_N,row_el):
    mat0 = np.zeros((input_N,1))
    mat1 = np.ones((input_N,1))
    if row_el == (0,0):
      return np.concatenate((mat0,mat0),axis = 1)
    elif row_el == (0,1):
      return np.concatenate((mat0,mat1),axis = 1)
    elif row_el == (1,0):
      return np.concatenate((mat1,mat0),axis = 1)
    elif row_el == (1,1):
      return np.concatenate((mat1,mat1),axis = 1)

  def get_logic_input_batch(self):
    return self.X

  def get_logic_output_batch(self,operation):
    mat1,mat2 = np.split(self.X,[1],axis=1)
    result_Y = np.zeros((self.N,1))
    if operation == "and":
      result_Y = mat1 * mat2
    elif operation == "or":
      for i in range(self.N):
        if mat1[i][0] == 0 and mat2[i][0] == 0:
          result_Y[i][0] = 0
        else:
          result_Y[i][0] = 1
    
    elif operation == "xor":
      for i in range(self.N):
        if mat1[i][0] == mat2[i][0]:
          result_Y[i][0] = 0
        else:
          result_Y[i][0] = 1
    result_Y = np.concatenate((self.X,result_Y),axis = 1)
    return result_Y

  def get(self):
    return self.Y

  def add_noise(self, std):
    for i in range(self.N * 4):
        for a in range(3):
            make_noise = np.random.normal()
            set_noise = std * make_noise
            self.Y[i][a] = round(self.Y[i][a] + set_noise,2)
    return self.Y

  def shuffle(self):
    X1,X2 = np.split(self.X,[1],axis=0)
    self.X = np.concatenate((X2,X1), axis= 0)
    Y1,Y2 = np.split(self.Y,[1],axis=0)
    self.Y = np.concatenate((Y2,Y1), axis= 0)

d = LogicDataset(10, "and")
x = d.get()
print(x,x.shape)
d.add_noise(0.1)
print(d.get())
d.shuffle()
print(d.get())


#%%
# solution of problem NumPy & Matplotlib 2

W = np.array([[4, -7],[5, 3]])
xi = np.arange(0,2.1,0.1) -1
x1 = xi.reshape((21,1))
x2 = x1
xx = np.concatenate((x1,x2), axis= 1)
Z= np.dot(xx,W)

Z1,Z2 = np.split(Z,[1],axis=1)
imsi = Z1 + Z2
output = 1/(1 + np.exp(-imsi))

plt.imshow(output)
plt.show()


#%%
# solution of problem NumPy & Matplotlib 3


fx_x = np.arange(0,20.1,0.1)-10

fx_h = np.exp(1) - 8
fx_w = 0.25 * math.pi

fx_a_y = ((((fx_x + fx_h) * (fx_x + fx_h) * (fx_x + fx_h)) - ((fx_x + fx_h) * (fx_x + fx_h)) - ((fx_x + fx_h) * 50) + 1) - (((fx_x - fx_h) * (fx_x - fx_h) * (fx_x - fx_h)) - ((fx_x - fx_h) * (fx_x - fx_h)) - ((fx_x - fx_h) * 50) + 1)) / (2 * fx_h) 
fx_b_y = (np.sin(fx_x * fx_w + fx_h) - np.sin(fx_x * fx_w - fx_h)) / (2*fx_h)
fx_c_y = (1/(1 + np.exp(-(fx_x + fx_h))) - 1/(1 + np.exp(-fx_x - fx_h)))/ (2*fx_h)
fx_d_y = (np.tanh(fx_x + fx_h) - np.tanh(fx_x - fx_h)) / (2*fx_h)

plt.plot(fx_x, fx_a_y)
plt.show()
plt.plot(fx_x, fx_b_y)
plt.show()
plt.plot(fx_x, fx_c_y)
plt.show()
plt.plot(fx_x, fx_d_y)
plt.show()
