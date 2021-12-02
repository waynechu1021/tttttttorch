from scipy.io import loadmat
import loss
import optim
import nn
import numpy as np
import matplotlib.pyplot as plt
from tensor import Tensor

def accuracy(a, y):
    size = a.shape[0]
    idx_a = np.argmax(a, axis=1)
    idx_y = np.argmax(y, axis=1)
    acc = sum(idx_a == idx_y) / size
    return acc

trainData = np.load("example/train_x.npy")
trainLabels = np.load("example/train_y.npy").T
testData = np.load("example/test_x.npy")
testLabels = np.load("example/test_y.npy").T
'''
trainData = Tensor(trainData)
trainLabels = Tensor(trainLabels)
testData = Tensor(testData)
testLabels = Tensor(testLabels)
'''
train_size = 20000
train_x = trainData.reshape(-1, train_size).T
train_x = train_x.reshape(train_size, 1, 28, 28) / 255.
test_size = 5000
test_x = testData.reshape(-1, test_size).T
test_x = test_x.reshape(test_size, 1, 28, 28) / 255.

class Net(nn.Module):
    def __init__(self):
        self.Sequential = nn.Sequential(
                            nn.Conv2d(1,6,(5,5),stride = 1,padding = 2),    #N*1*28*28 -> N*6*28,28
                            nn.BatchNorm2d(6),
                            nn.Maxpool(2,2),                   #N*6*14,14 -> N*6*14*14   
                            nn.Relu(),        
                            nn.Conv2d(6,16,(5,5),stride = 1,padding = 2),              #N*6*14,14 -> N*16*14*14
                            nn.BatchNorm2d(16),
                            nn.Maxpool(2,2),                   #N*16*14*14 -> N*16*7*7
                            nn.Relu(),
                            nn.Linear(784,256),
                            nn.Dropout(),
                            nn.Relu(),
                            nn.Linear(256,10),
                            nn.SoftMax())
        super(Net, self).__init__(self.Sequential)

model = Net()
optimizer = optim.Adam(model, lr=0.001)
criterion = loss.CrossEntropy()

E_list = []
Loss_list = []
acc_list = []
batch_size = 128
max_epoch = 10
for epoch in range(max_epoch):
    E_list.append(epoch)
    # Forward 前向传播
    sample_idxs = np.random.permutation(train_x.shape[0])
    # just for test autograd
    #sample_idxs = np.arange(train_x.shape[0])
    num_batch = int(np.ceil(train_x.shape[0]/batch_size))
    train_cost = 0
    for batch_idx in range(num_batch):
            x = Tensor(train_x[sample_idxs[batch_size*batch_idx:min(batch_size*(batch_idx + 1),train_x.shape[0])],:,:,:])
            y = Tensor(trainLabels[sample_idxs[batch_size*batch_idx:min(batch_size*(batch_idx + 1),trainLabels.shape[0])],:])
            y_pred = model(x)
            optimizer.zero_grad()
            loss = criterion(y_pred,y)
            train_cost += loss.values / x.shape[0]
            loss.backward()
            optimizer.step()
    train_cost /= num_batch
    Loss_list.append(train_cost) #添加到loss列表中

    model.eval()
    test_pre = model(Tensor(test_x))
    test_pre = test_pre.values
    acc = accuracy(test_pre,testLabels)
    acc_list.append(acc)
    print("epoch= ",epoch," train cost = ",train_cost,"acc on testset = ",acc)
    model.train()


model.eval()
train_x = Tensor(train_x)
trainLabels = Tensor(trainLabels)
train_pre = model(train_x)
train_pre = train_pre.values
trainLabels = trainLabels.values
train_acc = accuracy(train_pre,trainLabels)
print("Acuuracy on train set:",train_acc)

test_x = Tensor(test_x)
testLabels = Tensor(testLabels)
test_pre = model(test_x)
test_pre = test_pre.values
testLabels = testLabels.values
test_acc = accuracy(test_pre,testLabels)
print("Acuuracy on test set:",test_acc)