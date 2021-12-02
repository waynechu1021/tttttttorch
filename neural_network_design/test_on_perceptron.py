import nn
import numpy as np
from matplotlib import pyplot as plt


train_x = []
train_y = []
def show(train_x, train_y, w, b):
    x1 = np.arange(-10, 10, 1)
    x2 = (w[0] * x1 + b) / (-w[1])
    for i in range(len(train_x)):
        if(train_y[i] == 1):
            plt.scatter(train_x[i][0], train_x[i][1], color='red')
        else:
            plt.scatter(train_x[i][0], train_x[i][1], color='blue')
    plt.plot(x1, x2)
    plt.show()

if __name__ == '__main__':
    np.random.seed(0)
    X=np.r_[np.random.randn(20,2)-[3,3],np.random.randn(20,2)+[3,3]]
    Y=20*[1]+20*[-1]
    for i in range(len(X)):
        train_x.append(X[i])
        train_y.append(Y[i])

    perc=nn.perceptron()
    w, b = perc(train_x,train_y)
    print("w: ", w, "b: ", b)
    show(train_x,train_y,w,b)
