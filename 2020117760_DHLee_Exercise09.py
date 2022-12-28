# Duho Lee

#Classifying the Name Nationality


import torch
import glob
import unicodedata
import string
import torch
import torch.nn as nn
import sys
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torch.autograd import Variable


##----데이터, 사전 매핑----##

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters)

def findFiles(path): return glob.glob(path)

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read a file and split into lines
def readLines(filename):
    lines = open(filename).read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

# Build the category_lines dictionary, a list of lines per category
category_lines = {}
all_categories = []
for filename in findFiles('data/names/*.txt'):
    category = filename.split('\\')[-1].split('.')[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)
print(category_lines['Italian'][:5])

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


##----모델----##

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden): #한 단계 실행, input은 (1 x n_letters)의 텐서 즉 한 글자의 one-hot 벡터, hidden은 (1 x hidden_size)의 텐서,이전 단계의 hidden state
        combined = torch.cat((input, hidden), 1)  #torch.cat은 텐서를 연결해주는 함수, 1은 열방향으로 연결하라는 의미, 연결하는 이유는 hidden state와 input을 결합해야 하기 때문
        hidden = self.i2h(combined) #hidden state를 계산, nn.Linear는 선형변환을 해주는 함수, input_size + hidden_size는 입력의 차원, hidden_size는 출력의 차원
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden #hidden은 다음 단계의 hidden state로 사용, output은 softmax를 거친 확률분포,output은 (1 x n_categories)의 텐서, hidden은 (1 x hidden_size)의 텐서

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

##학습##

n_hidden = 128
n_epochs = 100000
print_every = 5000
plot_every = 1000
learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

def categoryFromOutput(output): #output은 evaluate의 결과, 즉 output의 해석
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    category_i = top_i[0][0]
    return all_categories[category_i], category_i

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(lineToTensor(line))
    return category, line, category_tensor, line_tensor

rnn = RNN(n_letters, n_hidden, n_categories)  #input,hidden,output
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

def train(category_tensor, line_tensor): #입력은 category와 line의 one-hot 벡터,line은 단어(성)의 길이만큼의 벡터
    hidden = rnn.initHidden() #hidden state 초기화, 처음에는 0으로 초기화
    optimizer.zero_grad() #zero_grad()는 gradient를 0으로 초기화

    for i in range(line_tensor.size()[0]): #line_tensor.size()의 결과는 (단어의 길이, 1, n_letters)이므로 line_tensor.size()[0]은 단어의 길이
        output, hidden = rnn(line_tensor[i], hidden) 

    loss = criterion(output, category_tensor) #for문의 최종 output과 카테고리를 비교
    loss.backward()

    optimizer.step()

    return output, loss.data

# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for epoch in range(1, n_epochs + 1):
    category, line, category_tensor, line_tensor = randomTrainingPair()
    output, loss = train(category_tensor, line_tensor) #각 line(단어)마다 loss를 계산
    current_loss += loss

    # Print epoch number, loss, name and guess
    if epoch % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if epoch % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

torch.save(rnn, 'char-rnn-classification.pt')

plt.figure()
plt.plot(all_losses)

#rnn의 parameter를 출력
print(rnn)

##-----predict,사용자입력으로 실행----##

rnn = torch.load('char-rnn-classification.pt')

# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.initHidden()
    
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    
    return output

def predict(line, n_predictions=3):
    output = evaluate(Variable(lineToTensor(line)))

    # Get top N categories
    topv, topi = output.data.topk(n_predictions, 1, True)
    predictions = []

    for i in range(n_predictions):
        value = topv[0][i]
        category_index = topi[0][i]
        print('(%.2f) %s' % (value, all_categories[category_index]))
        predictions.append([value, all_categories[category_index]])

    return predictions

print(predict("Dovesky"))
print(predict("Satoshi"))
print(predict("Park"))
print(predict("Jackson"))


#%%
# solution of Homework #2
#Modify the tutorial code to use nn.LSTM 


import torch
import glob
import unicodedata
import string
import torch
import torch.nn as nn
import sys
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torch.autograd import Variable


##----데이터, 사전 매핑----##

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters)

def findFiles(path): return glob.glob(path)

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read a file and split into lines
def readLines(filename):
    lines = open(filename).read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

# Build the category_lines dictionary, a list of lines per category
category_lines = {}
all_categories = []
for filename in findFiles('data/names/*.txt'):
    category = filename.split('\\')[-1].split('.')[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)
print(category_lines['Italian'][:5])

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


##----모델----##

class MyLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
      super().__init__()

      self.input_size= input_size
      self.hidden_size= hidden_size
      self.ih= nn.Linear(input_size, 4* hidden_size)  #
      self.hh= nn.Linear(hidden_size, 4* hidden_size)

    def forward(self, input, hc): #input은 line_tensor[i], hc는 (hn, cn)
      hx, cx= hc
      
      self.z= self.ih(input) + self.hh(hx)

      gi, gf, gc, go= torch.chunk(self.z, 4, dim=1)
      
      i= torch.sigmoid(gi)
      f= torch.sigmoid(gf)
      c= torch.tanh(gc)
      o= torch.sigmoid(go)
      
      c_n= f* cx+ i* c
      h_n= o* torch.tanh(c_n)
      
      return h_n, c_n

##학습##

n_hidden = 128
n_epochs = 100000
print_every = 5000
plot_every = 1000
learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

def categoryFromOutput(output): #output은 evaluate의 결과, 즉 output의 해석
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    category_i = top_i[0][0]
    return all_categories[category_i], category_i

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(lineToTensor(line))
    return category, line, category_tensor, line_tensor

rnn = MyLSTMCell(n_letters, n_hidden)
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

def train(category_tensor, line_tensor): #입력은 category와 line의 one-hot 벡터,line은 단어(성)의 길이만큼의 벡터
    output = []
    hn= Variable(torch.zeros(1, n_hidden))
    cn= Variable(torch.zeros(1, n_hidden))

    optimizer.zero_grad() #zero_grad()는 gradient를 0으로 초기화

    for i in range(line_tensor.size()[0]): #line_tensor.size()의 결과는 (단어의 길이, 1, n_letters)이므로 line_tensor.size()[0]은 단어의 길이
        (hn, cn) = rnn(line_tensor[i],(hn, cn))
        output.append(hn)

    output = torch.stack(output)
    output = output[-1].view(1, n_hidden)
    output = nn.Linear(n_hidden, n_categories)(output)

    loss = criterion(output, category_tensor) #for문의 최종 output과 카테고리를 비교
    loss.backward()

    optimizer.step()

    return output, loss.data

# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for epoch in range(1, n_epochs + 1):
    category, line, category_tensor, line_tensor = randomTrainingPair()
    output, loss = train(category_tensor, line_tensor) #각 line(단어)마다 loss를 계산
    current_loss += loss

    # Print epoch number, loss, name and guess
    if epoch % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if epoch % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

torch.save(rnn, 'char-rnn-classification.pt')

plt.figure()
plt.plot(all_losses)

#rnn의 parameter를 출력
print(rnn)

##-----predict,사용자입력으로 실행----##

rnn = torch.load('char-rnn-classification.pt')

# Just return an output given a line
def evaluate(line_tensor):
    output = []
    hn= Variable(torch.zeros(1, n_hidden))
    cn= Variable(torch.zeros(1, n_hidden))

    for i in range(line_tensor.size()[0]): #line_tensor.size()의 결과는 (단어의 길이, 1, n_letters)이므로 line_tensor.size()[0]은 단어의 길이
        (hn, cn) = rnn(line_tensor[i],(hn, cn))
        output.append(hn)

    output = torch.stack(output)
    output = output[-1].view(1, n_hidden)
    output = nn.Linear(n_hidden, n_categories)(output)

    return output


def predict(line, n_predictions=3):
    output = evaluate(Variable(lineToTensor(line)))

    # Get top N categories
    topv, topi = output.data.topk(n_predictions, 1, True)
    predictions = []

    for i in range(n_predictions):
        value = topv[0][i]
        category_index = topi[0][i]
        print('(%.2f) %s' % (value, all_categories[category_index]))
        predictions.append([value, all_categories[category_index]])

    return predictions

print(predict("Dovesky"))
print(predict("Satoshi"))
print(predict("Park"))
print(predict("Jackson"))
