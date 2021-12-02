import gc
from tensor import Tensor
from abc import ABCMeta, abstractmethod
import numpy as np
import random
__all__ = ['Module']


class Layer(metaclass=ABCMeta):
    @abstractmethod
    def forward(self, inputs):
        pass

class Module():
    def __init__(self, Sequential):
        self.Sequential = Sequential
        self.mode = True

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return self.Sequential.forward(x, self.mode)

    def backward(self, output_delta):
        self.Sequential.backward(output_delta)


    def add_layer(self, layer):
        self.Sequential.add_layer(layer)

    def train(self):
        self.mode = True

    def eval(self):
        self.mode = False

class Sequential():
    def __init__(self, *layers):
        #super(Sequential,self).__init__()
        self.layer_list = []
        for layer in layers:
            self.layer_list.append(layer)

    def forward(self, x, mode):
        out = x
        for layer in self.layer_list:
            out = layer(out, mode)
        return out

    def backward(self, output_delta):
        layer_num = len(self.layer_list)
        delta = output_delta
        for i in range(layer_num - 1, -1, -1):
            # 反向遍历各个层, 将期望改变量反向传播
            delta = self.layer_list[i].backward(delta)

    def add_layer(self, layer):
        self.layer_list.append(layer)


class Conv2d(Layer):
    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel_size,
                 stride=1,
                 padding=0):
        super(Conv2d).__init__()
        assert isinstance(input_channel, int) and input_channel > 0
        assert isinstance(output_channel, int) and output_channel > 0
        assert isinstance(stride, int) and stride > 0
        assert isinstance(padding, int) and padding >= 0
        self.input_channel = input_channel
        self.output_channel = output_channel
        if isinstance(kernel_size, int):
            self.kernel_size_h = self.kernel_size_w = kernel_size
        else:
            assert len(kernel_size) == 2
            self.kernel_size_h = kernel_size[0]
            self.kernel_size_w = kernel_size[1]
        self.stride = stride
        if isinstance(padding, int):
            self.padding_h = self.padding_w = padding
        else:
            assert len(kernel_size) == 2
            self.padding_h = padding[0]
            self.padding_w = padding[1]
        self.kernel = Tensor.zeros((self.output_channel, self.input_channel,
                                self.kernel_size_h, self.kernel_size_w),requires_grad = True)
        self.bias = Tensor.zeros((1, output_channel),requires_grad = True)
        self.reset_parameters()

    def reset_parameters(self):
        bound = np.sqrt(6. / (self.output_channel + self.input_channel))
        #bound = 1
        #self.kernel = np.random.randn(self.output_channel, self.input_channel, self.kernel_size_h, self.kernel_size_w) / np.sqrt(self.output_channel / 2.)
        self.kernel = Tensor.uniform(-bound, bound, (self.output_channel, self.input_channel,self.kernel_size_h, self.kernel_size_w),requires_grad = True)
        #self.kernel = Tensor(np.arange(self.output_channel*self.input_channel*self.kernel_size_h*self.kernel_size_w).reshape((self.output_channel,self.input_channel,self.kernel_size_h,self.kernel_size_w)),requires_grad=True)
        #self.kernel = Tensor(np.ones((self.output_channel,self.input_channel,self.kernel_size_h,self.kernel_size_w))/10,requires_grad=True)
    def __call__(self, inputs, mode=True):
        return self.forward(inputs, mode)

    def forward(self, inputs, mode=True):
        z = Tensor.__conv__(inputs,self.kernel,self.bias,self.padding_h,\
                                self.padding_w,self.kernel_size_h,self.kernel_size_w,self.stride)
        return z


class Maxpool(Layer):
    def __init__(self, size, stride=1):
        self.size = size  # maxpool框的尺寸
        self.stride = stride

    def __call__(self, inputs, mode=True):
        return self.forward(inputs, mode)

    def forward(self, inputs, mode=True):
        #self.inputs = inputs
        return Tensor.__maxpool__(inputs,self.size,self.stride)
         


class BatchNorm(Layer):
    def __init__(self,num_features,eps = 1e-5,momentum = 0.1):
        super(BatchNorm).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.reset_parameters()
    def reset_parameters(self):
        self.gamma = Tensor(np.ones((1,self.num_features)),requires_grad=True)
        self.beta = Tensor(np.zeros((1,self.num_features)),requires_grad=True)
        self.running_mean = np.zeros((1, self.num_features))
        self.running_var = np.ones((1, self.num_features))
    def __call__(self, inputs,mode = True):
        return self.forward(inputs,mode)
    def forward(self,inputs,mode):
        if mode:
            self.x_mean = np.mean(inputs.values,axis = 0,keepdims=True)        #均值
            self.x_var = np.var(inputs.values,axis = 0,keepdims=True)          #方差
            self.running_mean = (1-self.momentum)*self.x_mean + self.momentum*self.running_mean
            self.running_var = (1-self.momentum)*self.x_var + self.momentum*self.running_var
            y = (inputs-self.x_mean)/(self.x_var+self.eps)**0.5
            return self.gamma*y + self.beta
        else:
            y = (inputs - self.running_mean) / (self.running_var+self.eps)**0.5
            return self.gamma*y + self.beta



class BatchNorm2d(Layer):
    def __init__(self,num_features,eps = 1e-5,momentum = 0.1):
        super(BatchNorm2d).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.reset_parameters()
    def reset_parameters(self):
        self.gamma = Tensor(np.ones((1,self.num_features,1,1)),requires_grad=True)
        self.beta = Tensor(np.zeros((1,self.num_features,1,1)),requires_grad=True)
        self.running_mean = np.zeros((1, self.num_features, 1, 1))
        self.running_var = np.ones((1, self.num_features, 1, 1))
    def __call__(self, inputs,mode = True):
        return self.forward(inputs,mode)
    def forward(self,inputs,mode):
        # inputs.shape == [N,C,height,width]
        if mode:
            self.x_mean = np.mean(inputs.values,axis = (0,2,3),keepdims=True)        #均值
            self.x_var = np.var(inputs.values,axis = (0,2,3),keepdims=True)          #方差
            self.running_mean = (1-self.momentum)*self.x_mean + self.momentum*self.running_mean
            self.running_var = (1-self.momentum)*self.x_var + self.momentum*self.running_var
            y = (inputs-self.x_mean)/(self.x_var+self.eps)**0.5
            return self.gamma*y + self.beta
        else:
            y = (inputs - self.running_mean) / (self.running_var+self.eps)**0.5
            return self.gamma*y + self.beta


class Dropout(Layer):
    def __init__(self,p = 0.2):
        #p指的是被丢弃的概率
        self.p = p
        self.mask = None
    def __call__(self,inputs,mode = True):
        return self.forward(inputs,mode)
    def forward(self,inputs,mode):
        c = 1 - self.p
        if mode:
            self.mask = Tensor(np.random.binomial(1,1 - self.p,inputs.shape))
            c = self.mask
        return inputs*c

class Sigmoid(Layer):
    def __init__(self):
        pass

    def __call__(self, s, mode=True):
        return self.forward(s, mode)

    def forward(self, s, mode=True):
        return self.func(s)

    def func(self, s):
        return 1 / (1 + Tensor.exp(-s))

class Relu(Layer):
    def __init__(self):
        pass

    def __call__(self, s, mode=True):
        return self.forward(s, mode)

    def forward(self, s, mode=True):
        return self.func(s)

    def func(self, s):
        return s * (s > 0)



class SoftMax(Layer):
    def __init__(self):
        pass

    def __call__(self, s, mode=True):
        return self.forward(s, mode)

    def forward(self, s, mode=True):
        return self.func(s)

    def func(self, s):
        #max = Tensor.__max__(s.values)
        #return Tensor.exp(s - max) / Tensor.sum(Tensor.exp(s - max), axis=1, keepdims=True)
        return Tensor.__softmax__(s)
    

class Linear(Layer):
    def __init__(self, num_in, num_out):
        super(Linear, self).__init__()
        assert isinstance(num_in, int) and num_in > 0
        assert isinstance(num_out, int) and num_out > 0
        self.num_in = num_in
        self.num_out = num_out
        self.weight = Tensor.zeros((num_in, num_out),requires_grad= True)
        self.bias = Tensor.zeros((1, num_out),requires_grad= True)
        self.reset_parameters()

    def __call__(self, inputs, mode = True):
        return self.forward(inputs, mode)

    def reset_parameters(self):
        bound = np.sqrt(6. / (self.num_in + self.num_out))
        self.weight = Tensor.uniform(-bound, bound,(self.num_in, self.num_out),requires_grad= True)
        #self.weight = Tensor(np.ones((self.num_in, self.num_out))/100,requires_grad=True)
        del bound
        gc.collect()

    def forward(self, inputs, mode = True):
        # inputs.shape == [N,num_in]
        
        inputs = inputs.reshape((inputs.shape[0], -1))
        assert len(inputs.shape) == 2 and inputs.shape[1] == self.num_in
        assert self.weight.shape == (self.num_in, self.num_out)
        assert self.bias.shape == (1, self.num_out)

        z = Tensor.__matmul__(inputs, self.weight) + self.bias
        return z


class perceptron():
    def __init__(self):
        '''
         :param w:感知机的权重
         :param b:感知机的偏置
         :param learning_rate:学习率
        '''
        self.w = np.array([0, 0])
        self.b = 0
        self.learning_rate = 0.5

    def __call__(self,train_x,train_y):
        return self.train(train_x,train_y)

    def update(self, w, x, y, b):
        '''
        该函数用于参数的更新
        :param w: 权重
        :param x: 数据的特征
        :param y: 数据的标签
        :param b: 数据的偏置
        :return: 无
        '''
        self.w = w+np.multiply(self.learning_rate,x)*y
        self.b = b+self.learning_rate*y

    def sign(self, w, x, b):
        '''
        该部分为符号函数
        :return 返回计算后的符号函数的值
        '''
        return np.sign(np.dot(w, x)+b)

    def train(self, train_x, train_y):
        '''
        该函数使用随机选择数据点来进行训练（随机梯度下降法）
        :param data: 输入数据
        :return: 返回最终训练好模型（参数）
        '''
        stop = True
        while stop:
            count = len(train_x)
            index = [i for i in range(len(train_x))]
            random.shuffle(index)
            for i in index:
                if self.sign(self.w, train_x[i], self.b)*train_y[i] <= 0:
                    self.update(self.w, train_x[i], train_y[i], self.b)
                else:
                    count -= 1
            if count == 0:
                stop = False
        print("最终w:", self.w, "最终b:", self.b)
        return self.w, self.b