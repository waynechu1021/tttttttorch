from abc import ABCMeta, abstractmethod
import nn
import numpy as np
import math

class Optim(metaclass=ABCMeta):
    @abstractmethod
    def step(self):
        pass


class SGD(Optim):
    def __init__(self, Module, lr = 1e-3, lr_decay=0):
        self.Module = Module
        self.learning_rate = lr
        self.lr_decay = lr_decay
        if lr_decay:
            self.epoch = 0

    def step(self):
        if self.lr_decay:
            self.epoch += 1
            self.learning_rate = self.learning_rate * (1 - self.lr_decay)**(
                self.epoch // 100)
        else:
            self.learning_rate = self.learning_rate * 1
        
        for layer in self.Module.Sequential.layer_list:
            if isinstance(layer,nn.Linear) == True:
                layer.weight.values = layer.weight.values - self.learning_rate*layer.weight.grad
                layer.bias.values = layer.bias.values - self.learning_rate * layer.bias.grad
            if isinstance(layer,nn.Conv2d) == True:
                layer.kernel.values = layer.kernel.values - self.learning_rate*layer.kernel.grad
                layer.bias.values = layer.bias.values - self.learning_rate * layer.bias.grad
            if isinstance(layer,nn.BatchNorm2d) ==True:
                layer.gamma.values = layer.gamma.values - self.learning_rate*layer.gamma.grad
                layer.beta.values = layer.beta.values - self.learning_rate*layer.beta.grad
    def zero_grad(self):
        for layer in self.Module.Sequential.layer_list:
            if isinstance(layer,nn.Linear) == True:
                layer.weight.zero_grad()
                layer.bias.zero_grad()
            if isinstance(layer,nn.Conv2d) == True:
                layer.kernel.zero_grad()
                layer.bias.zero_grad()
            if isinstance(layer,nn.BatchNorm2d) ==True:
                layer.gamma.zero_grad()
                layer.beta.zero_grad()

class Adam(Optim):
    def __init__(self, Module, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,lr_decay=0):
        self.Module = Module
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.lr_decay = lr_decay
        self.m = None
        self.v = None
        self.t = 0
        
    def step(self):
        if self.m is None:
            self.m, self.v = {}, {}
        beta1, beta2 = self.betas
        for layer in self.Module.Sequential.layer_list: 
            if isinstance(layer,nn.Linear) == True:
                self.m["weight"] = np.zeros(layer.weight.grad.shape)
                self.v["weight"] = np.zeros(layer.weight.grad.shape)

                self.m["bias"] = np.zeros(layer.bias.grad.shape)
                self.v["bias"] = np.zeros(layer.bias.grad.shape)
                self.t += 1
                lr_t = self.lr * math.sqrt(1.0 - beta2**self.t) / (1.0 - beta1**self.t)

                self.m["weight"] += (1 - beta1) * (layer.weight.grad - self.m["weight"])
                self.v["weight"] += (1 - beta2) * (layer.weight.grad**2 - self.v["weight"])
                temp_weight = lr_t * self.m["weight"] / (np.sqrt(self.v["weight"]) + self.eps)
                layer.weight.values = layer.weight.values - temp_weight

                self.m["bias"] += (1 - beta1) * (layer.bias.grad - self.m["bias"])
                self.v["bias"] += (1 - beta2) * (layer.bias.grad**2 - self.v["bias"])
                temp_bias = lr_t * self.m["bias"] / (np.sqrt(self.v["bias"]) + self.eps)
                layer.bias.values = layer.bias.values - temp_bias

            if isinstance(layer,nn.Conv2d) == True:
                self.m["kernel"] = np.zeros(layer.kernel.grad.shape)
                self.v["kernel"] = np.zeros(layer.kernel.grad.shape)

                self.m["bias"] = np.zeros(layer.bias.grad.shape)
                self.v["bias"] = np.zeros(layer.bias.grad.shape)
                self.t += 1
                lr_t = self.lr * math.sqrt(1.0 - beta2**self.t) / (1.0 - beta1**self.t)
                self.m["kernel"] += (1 - beta1) * (layer.kernel.grad - self.m["kernel"])
                self.v["kernel"] += (1 - beta2) * (layer.kernel.grad**2 - self.v["kernel"])
                layer.kernel.values = layer.kernel.values - lr_t * self.m["kernel"] / (np.sqrt(self.v["kernel"]) + self.eps)

                self.m["bias"] += (1 - beta1) * (layer.bias.grad - self.m["bias"])
                self.v["bias"] += (1 - beta2) * (layer.bias.grad**2 - self.v["bias"])
                layer.bias.values = layer.bias.values - lr_t * self.m["bias"] / (np.sqrt(self.v["bias"]) + self.eps)

            if isinstance(layer,nn.BatchNorm2d) == True or isinstance(layer,nn.BatchNorm) == True:
                self.m["gamma"] = np.zeros(layer.gamma.grad.shape)
                self.v["gamma"] = np.zeros(layer.gamma.grad.shape)

                self.m["beta"] = np.zeros(layer.beta.grad.shape)
                self.v["beta"] = np.zeros(layer.beta.grad.shape)
                self.t += 1
                lr_t = self.lr * math.sqrt(1.0 - beta2**self.t) / (1.0 - beta1**self.t)

                self.m["gamma"] += (1 - beta1) * (layer.gamma.grad - self.m["gamma"])
                self.v["gamma"] += (1 - beta2) * (layer.gamma.grad**2 - self.v["gamma"])
                temp_gamma = lr_t * self.m["gamma"] / (np.sqrt(self.v["gamma"]) + self.eps)
                layer.gamma.values = layer.gamma.values - temp_gamma

                self.m["beta"] += (1 - beta1) * (layer.beta.grad - self.m["beta"])
                self.v["beta"] += (1 - beta2) * (layer.beta.grad**2 - self.v["beta"])
                temp_beta = lr_t * self.m["beta"] / (np.sqrt(self.v["beta"]) + self.eps)
                layer.beta.values = layer.beta.values - temp_beta
    def zero_grad(self):
        for layer in self.Module.Sequential.layer_list:
            if isinstance(layer,nn.Linear) == True:
                layer.weight.zero_grad()
                layer.bias.zero_grad()
            if isinstance(layer,nn.Conv2d) == True:
                layer.kernel.zero_grad()
                layer.bias.zero_grad()
            if isinstance(layer,nn.BatchNorm2d) ==True:
                layer.gamma.zero_grad()
                layer.beta.zero_grad()