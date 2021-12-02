from scipy.io import loadmat
import loss
import optim
import nn
import numpy as np
import matplotlib.pyplot as plt
from evaluator import Confusion_Matrix, Performance_Index
from tensor import Tensor
import pandas as pd

def accuracy(a, y):
    size = a.shape[0]
    idx_a = np.argmax(a, axis=1)
    idx_y = np.argmax(y, axis=1)
    acc = sum(idx_a==idx_y) /size
    return acc

def draw_matrix(classes, con_mat, name):
    # 热度图，后面是指定的颜色块，gray也可以，gray_x反色也可以
    plt.imshow(con_mat, cmap=plt.cm.Blues)
    # ticks 这个是坐标轴上的坐标点
    # label 这个是坐标轴的注释说明
    indices = range(len(con_mat))
    # 坐标位置放入
    # 第一个是迭代对象，表示坐标的顺序
    # 第二个是坐标显示的数值的数组，第一个表示的其实就是坐标显示数字数组的index，但是记住必须是迭代对象
    plt.xticks(indices, classes)
    plt.yticks(indices, classes)
    # 热度显示仪
    plt.colorbar()
    plt.xlabel('real')
    plt.ylabel('pre')
    # 显示数据，直观些
    for first_index in range(len(con_mat)):
        for second_index in range(len(con_mat[first_index])):
            plt.text(first_index, second_index, con_mat[second_index][first_index])
    # 显示
    plt.savefig(name)
    plt.show()

m = loadmat("./example/mnist_small_matlab.mat")
#trainData:28*28*10000    trainlabels:10*10000 
trainData, trainLabels = m['trainData'], m['trainLabels'].T
#testData:28*28*2000    testlabels:10*2000
testData, testLabels = m['testData'], m['testLabels'].T

train_size = 10000
train_x = trainData.reshape(-1,train_size).T #10000 * 784
train_x = train_x.reshape(10000,1,28,28)

test_size = 2000
test_x = testData.reshape(-1,test_size).T    #2000 * 784
test_x = test_x.reshape(2000,1,28,28)

class Net(nn.Module):
    def __init__(self):
        self.Sequential = nn.Sequential(nn.Linear(784,256),
                            nn.Relu(),        
                            nn.Linear(256,128),
                            nn.Relu(),        
                            nn.Linear(128,10),
                            nn.SoftMax())
        super(Net,self).__init__(self.Sequential)

model = Net()
optimizer = optim.Adam(model,lr=0.001)
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
train_pre = model(train_x)
train_pre = train_pre.values
train_acc = accuracy(train_pre,trainLabels)
print("Acuuracy on train set:",train_acc)

test_x = Tensor(test_x)
test_pre = model(test_x)
test_pre = test_pre.values
test_acc = accuracy(test_pre,testLabels)
print("Acuuracy on test set:",test_acc)

'''混淆矩阵'''
C_M = Confusion_Matrix(testLabels, test_pre)        # 实例化类对象
con_mat = C_M.con_mat()                             # 调用成员函数计算混淆矩阵
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]            # 类别
draw_matrix(classes, con_mat, "con_mat_Minist.png")   # 混淆矩阵画图
print(con_mat)
'''性能指标'''
per_index = Performance_Index(con_mat)                  # 实例化类对象
print("Acuuracy on train set:", train_acc)
print("Acuuracy on test set:", per_index.Acc_all())     # 计算整体准确率
acc = per_index.Acc()                                   # 计算每个类别分别的准确率
precision = per_index.Precision()                       # 计算每个类别分别的精确率
recall = per_index.Re_call()                            # 计算每个类别分别的召回率
data = {"precision": precision, "recall": recall, "Acc": acc}
df = pd.DataFrame(data, index=classes)                  # 使用DataFrame进行显示
print(df)