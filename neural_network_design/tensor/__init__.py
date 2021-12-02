"""
Tensor类及其重载的操作
"""
from math import fabs
import numpy as np 
import gc

#import sys  # 导入sys模块
#sys.setrecursionlimit(3000)  # 将默认的递归深度修改为3000

class DependencyNode():
    def __init__(self, tensor, grad_func):
        self.tensor = tensor
        self.grad_func = grad_func

class Tensor():
    def __init__(self, values, requires_grad=False, dependencies=[], dtype=None,grad = None):
        self._values = np.asarray(values, dtype)
        self.grad = None
        self.requires_grad = requires_grad
        if self.requires_grad:
            self.grad = grad
            if grad is None:
                self.grad = np.zeros(self.shape)
        
        self.dependencies = dependencies
    
    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, new_val):
        self._values = np.asarray(new_val)
        self.grad = None
    
    @property
    def shape(self):
        return self._values.shape

    @property
    def T(self):
        return self.transpose(axes=None)
    '''
    def reshape(self,*command,**map):
        self._values = self._values.reshape(*command,**map)
        if self.requires_grad:
            self.grad = self.grad.reshape(*command,**map)
        return Tensor(self._values,grad=self.grad,requires_grad=self.requires_grad,dependencies=self.dependencies)
    '''
    def reshape(self,new_shape):
        return reshape_(self,new_shape)
        
    @classmethod
    def pad(self,*command,**map):
        if command[0].requires_grad:
            return Tensor(np.pad(command[0].values,*command[1:],**map),requires_grad=command[0].requires_grad,grad = np.pad(command[0].grad,*command[1:]))
        else:
            return  Tensor(np.pad(command[0].values,*command[1:],**map),requires_grad=command[0].requires_grad)
    

    @classmethod
    def zeros(self,*command,**map):
        if "requires_grad" in map.keys() :
            requires_grad = map['requires_grad']
            del map['requires_grad']
        else:
            requires_grad = False
        return Tensor(np.zeros(*command,**map),requires_grad=requires_grad)
    
    @classmethod
    def sqrt(self,*command,**map):
        return Tensor(np.sqrt(*command,**map))
    
    @classmethod
    def uniform(self,*command,**map):
        if "requires_grad" in map.keys() :
            requires_grad = map['requires_grad']
            del map['requires_grad']
        else:
            requires_grad = False
        return Tensor(np.random.uniform(*command,**map),requires_grad=requires_grad)
    
    @classmethod
    def flip(self,*command,**map):
        return Tensor(np.flip(*command,**map))
    
    @classmethod
    def __max__(self,*command,**map):
        return Tensor(np.max(*command,**map))
    
    @classmethod
    def argmax(self,*command,**map):
        return Tensor(np.argmax(*command,**map))


    def zero_grad(self):
        self.grad = np.zeros(self.shape)
        #self.dependencies = []
    
    def backward(self, grad=None):
        """
        反向传播梯度
        """
        assert self.requires_grad == True
        
        gradient = 1.0 if grad is None else grad
        # 从求导开始到自身的梯度
        temp_grad = np.array(gradient)
        self.grad = self.grad + temp_grad

        for dependency in self.dependencies:
            dep_grad = dependency.grad_func(temp_grad)
            dependency.tensor.backward(dep_grad)
            gc.collect()
    
    # 以下为操作符的重载
    def __gt__(self, obj):
        return self.values > convert_to_tensor(obj).values
    
    def __lt__(self, obj):
        return self.values < convert_to_tensor(obj).values
    
    def __ge__(self, obj):
        return self.values >= convert_to_tensor(obj).values
    
    def __le__(self, obj):
        return self.values <= convert_to_tensor(obj).values

    def __add__(self, obj):
        return add_(self, convert_to_tensor(obj))
    
    def __radd__(self, obj):
        return add_(convert_to_tensor(obj), self)

    def __iadd__(self, obj):
        self.values = self.values + convert_to_tensor(obj).values
        return self

    def __sub__(self, obj):
        return sub_(self, convert_to_tensor(obj))

    def __rsub__(self, obj):
        return sub_(convert_to_tensor(obj), self)

    def __isub__(self, obj):
        self.values = self.values - convert_to_tensor(obj).values
        return self

    def __mul__(self, obj):
        return mul_(self, convert_to_tensor(obj))

    def __rmul__(self, obj):
        return mul_(convert_to_tensor(obj), self)

    def __imul__(self, obj):
        self.values = self.values * convert_to_tensor(obj).values
        return self

    def __truediv__(self, obj):
        return div_(self, convert_to_tensor(obj))

    def __rtruediv__(self, obj):
        return div_(convert_to_tensor(obj), self)

    def __itruediv__(self, obj):
        self.values = self.values / convert_to_tensor(obj).values
        return self

    def __neg__(self):
        return neg_(self)

    def __getitem__(self, key):
        return getitem_(self, key)

    def __pow__(self, obj):
        return pow_(self, convert_to_tensor(obj))

    def __rpow__(self, obj):
        return pow_(convert_to_tensor(obj), self)

    def __ipow__(self, obj):
        self.values = self.values ** convert_to_tensor(obj).values
        return self

    def __matmul__(self, obj):
        return matmul_(self, convert_to_tensor(obj))

    def __rmatmul__(self, obj):
        return matmul_(convert_to_tensor(obj), self)

    def __imatmul__(self, obj):
        self.values = self.values @ convert_to_tensor(obj).values
        return self
    
    def __len__(self):
        return len(self.values)
    
    def __conv__(self,kernel_t,bias_t,padding_h,padding_w,kernel_size_h,kernel_size_w,stride):
        return conv_(self,kernel_t,bias_t,padding_h,padding_w,kernel_size_h,kernel_size_w,stride)

    def __maxpool__(self,size,stride):
        return maxpool_(self,size,stride)
    
    def __softmax__(self):
        return softmax__(self)

    # 以下为对numpy中方法的重载
    def transpose(self, axes=None):
        return transpose_(self, axes=axes)

    def log(self):
        return log_(self)

    def sin(self):
        return sin_(self)
    
    def exp(self):
        return exp_(self)

    def sum(self, axis=None, keepdims=False):
        return sum_(self, axis=axis, keepdims=keepdims)

def convert_to_tensor(obj, requires_grad=False):
    """
    将一个数或者numpy数组转化为Tensor
    """
    if not isinstance(obj, Tensor):
        obj = Tensor(obj, requires_grad=requires_grad)
    return obj

#==========================================================================================   
# utils
def create_binary_ops_tensor(values, tensor1, tensor2, grad_func_ts1, grad_func_ts2):
    """
    两个操作数形成的tensor (一个计算图上的结点)
    """
    requires_grad = tensor1.requires_grad or tensor2.requires_grad
    dependencies = []
    if tensor1.requires_grad:
        dependencies.append(DependencyNode(tensor=tensor1, grad_func=grad_func_ts1))
    if tensor2.requires_grad:
        dependencies.append(DependencyNode(tensor=tensor2, grad_func=grad_func_ts2))
    
    return Tensor(values, requires_grad, dependencies)

def create_unary_op_tensor(values, tensor1, grad_func_ts1):
    """
    一个操作数形成的tensor (一个计算图上的结点)
    """
    dependencies = []
    if tensor1.requires_grad:
        dependencies.append(DependencyNode(tensor=tensor1, grad_func=grad_func_ts1))
    return Tensor(values, tensor1.requires_grad, dependencies)
def create_trinary_op_tensor(values,tensor1,tensor2,tensor3,grad_func_ts1,grad_func_ts2,grad_func_ts3):
    requires_grad = tensor1.requires_grad or tensor2.requires_grad or tensor3.requires_grad
    dependencies = []
    if tensor1.requires_grad:
        dependencies.append(DependencyNode(tensor=tensor1, grad_func=grad_func_ts1))
    if tensor2.requires_grad:
        dependencies.append(DependencyNode(tensor=tensor2, grad_func=grad_func_ts2))
    if tensor3.requires_grad:
        dependencies.append(DependencyNode(tensor=tensor3, grad_func=grad_func_ts3))
    return Tensor(values, requires_grad, dependencies)
def avoid_broadcasting(grad, tensor):
    """
    防止因为broadcasting引起的矩阵尺寸变化, 进而导致传递出现问题
    """
    
    nidm = grad.ndim
    for _ in range(nidm - tensor.values.ndim):
        # 将grad的维度降到与tensor1的维度一致, 防止出现broadcasting
        grad = grad.sum(axis=0) 
    for i, dim in enumerate(tensor.shape):
        if dim == 1:
            # 如果tensor的某一维数值为1, 则grad按该维相加, 但是保持维度特性. 
            # 也是防止出现broadcasting
            grad = grad.sum(axis=i, keepdims=True)
    return grad  

#==========================================================================================
# operations
def reshape_(Tensor,new_shape):
    shape = Tensor.shape
    Tensor._values = Tensor._values.reshape(new_shape)
    def grad_func(grad):
        return grad.reshape(shape)
    return create_unary_op_tensor(Tensor.values,Tensor,grad_func)

def add_(tensor1, tensor2):
    values = tensor1.values + tensor2.values
    
    def grad_func_ts1(grad):
        grad = grad * 1.0
        return avoid_broadcasting(grad, tensor1)
    
    def grad_func_ts2(grad):
        grad = grad * 1.0
        return avoid_broadcasting(grad, tensor2)
    
    return create_binary_ops_tensor(values, tensor1, tensor2, grad_func_ts1, 
                                    grad_func_ts2)

def sub_(tensor1, tensor2):
    return tensor1 + (-tensor2)

def mul_(tensor1, tensor2):
    values = tensor1.values * tensor2.values

    def grad_func_ts1(grad):
        #grad = grad.reshape(tensor2.values.shape)
        grad = grad * tensor2.values #不能写成 grad *= tensor2.values
        return avoid_broadcasting(grad, tensor1)

    def grad_func_ts2(grad):
        #grad = grad.reshape(tensor1.values.shape)
        grad = grad * tensor1.values
        return avoid_broadcasting(grad, tensor2)
    
    return create_binary_ops_tensor(values, tensor1, tensor2, grad_func_ts1,
                                    grad_func_ts2)

def div_(tensor1, tensor2):
    values = tensor1.values / tensor2.values

    def grad_func_ts1(grad):
        grad = grad / tensor2.values
        return avoid_broadcasting(grad, tensor1)
    
    def grad_func_ts2(grad):
        grad = -grad * tensor1.values / tensor2.values ** 2
        return avoid_broadcasting(grad, tensor2)
    
    return create_binary_ops_tensor(values, tensor1, tensor2, grad_func_ts1,
                                    grad_func_ts2)
                
def pow_(tensor1, tensor2):
    values = tensor1.values ** tensor2.values

    def grad_func_ts1(grad):
        grad = grad * tensor2.values * tensor1.values ** (tensor2.values - 1)
        return avoid_broadcasting(grad, tensor1)
    
    def grad_func_ts2(grad):
        grad = grad * np.log(tensor1) * values
        return avoid_broadcasting(grad, tensor2)

    return create_binary_ops_tensor(values, tensor1, tensor2, grad_func_ts1,
                                    grad_func_ts2)
    
def matmul_(tensor1, tensor2):
    
    values = tensor1.values @ tensor2.values
    
    def grad_func_ts1(grad):
        return grad @ tensor2.values.T 
    
    def grad_func_ts2(grad):
        return tensor1.values.T @ grad 
    
    return create_binary_ops_tensor(values, tensor1, tensor2, grad_func_ts1,
                                    grad_func_ts2)
def im2col(img,kernel_size,stride=1,padding=0):
    #img.shape = [batch,channel,height,weight]
    N,C,H,W = img.shape
    if isinstance(kernel_size, int):
        kernel_size_h = kernel_size_w = kernel_size
    else:
        assert len(kernel_size) == 2
        kernel_size_h = kernel_size[0]
        kernel_size_w = kernel_size[1]
    if isinstance(padding, int):
        padding_h = padding_w = padding
    else:
        assert len(kernel_size) == 2
        padding_h = padding[0]
        padding_w = padding[1]
    out_h = (H + 2 * padding_h - kernel_size_h)//stride + 1
    out_w = (W + 2 * padding_w - kernel_size_w)//stride + 1 
    #填充padiing  默认为0
    img = np.pad(img,[(0,0), (0,0), (padding_h, padding_h), (padding_w, padding_w)],'constant')
    col = np.zeros((N*out_h*out_w,C * kernel_size_h * kernel_size_w))
    for y in range(out_h):
        y_min = y * stride
        y_max = y_min + kernel_size_h
        y_start = y * out_w
        for x in range(out_w):
            x_min = x * stride
            x_max = x_min + kernel_size_w
            col[y_start+x::out_w * out_h, :] = img[:, :,y_min:y_max, x_min:x_max].reshape(N, -1)
    return col

def col2img(col,kernel_size,output_shape,stride=1):
    #col.shape = [N*out_h*out_w,kernel_size_h * kernel_size_w * C]
    N,C,H,W = output_shape
    if isinstance(kernel_size, int):
        kernel_size_h = kernel_size_w = kernel_size
    else:
        assert len(kernel_size) == 2
        kernel_size_h = kernel_size[0]
        kernel_size_w = kernel_size[1]
    assert col.shape[1] == C*kernel_size_h*kernel_size_w
    out_h = (H - kernel_size_h)//stride + 1
    out_w = (W - kernel_size_w)//stride + 1 
    assert col.shape[0] == N*out_h*out_w
    img = np.zeros(output_shape)
    for y in range(out_h):
        y_min = y * stride
        y_max = y_min + kernel_size_h
        y_start = y * out_w
        for x in range(out_w):
            x_min = x * stride
            x_max = x_min + kernel_size_w
            img[:, :,y_min:y_max, x_min:x_max] += col[y_start+x::out_w * out_h, :].reshape(N,C,kernel_size_h,kernel_size_w)
    return img

def conv_(img,kernel_t,bias_t,padding_h,padding_w,kernel_size_h,kernel_size_w,stride):
    #a = img.values
    inputs = img.values
    kernel = kernel_t.values
    bias = bias_t.values
    N,C,H,W = inputs.shape
    out_h = (H + 2 * padding_h - kernel_size_h)//stride + 1
    out_w = (W + 2 * padding_w - kernel_size_w)//stride + 1 
    temp_inputs = im2col(inputs,(kernel_size_h,kernel_size_w),stride,(padding_h,padding_w))
    temp_kernel = im2col(kernel,(kernel_size_h,kernel_size_w)).T
    # z.shape  [N*out_h*out_w,output_channel]
    z = np.dot(temp_inputs,temp_kernel) + bias
    z = z.reshape(N,out_h,out_w,-1).transpose(0,3,1,2)
    output_channel = z.shape[1]

    def grad_func_kernel(grad):
        grad = grad.reshape(z.shape)
        oh, ow = grad.shape[2:]
        #inputs = np.pad(a,[(0,0), (0,0), (padding_h, padding_h), (padding_w, padding_w)],'constant')
        tem_delta2 = grad.transpose(1,2,3,0).reshape(output_channel,-1)
        grad_w = tem_delta2 @ im2col(inputs,(kernel_size_h,kernel_size_w),stride,
                                        (padding_h,padding_w))\
                                        .reshape(N,out_h*out_w,C,kernel_size_h*kernel_size_w)\
                                            .transpose(2,3,1,0)\
                                                .reshape(kernel_size_h*kernel_size_w*C,out_h*out_w*N).T
        grad_w = grad_w.reshape(kernel.shape)
        assert grad_w.shape == kernel.shape
        return grad_w
    def grad_func_bias(grad):
        grad = grad.reshape(z.shape)
        grad_b = np.sum(grad,axis = (0,2,3),keepdims=True)
        grad_b = grad_b.reshape(grad_b.shape[0],grad_b.shape[1])
        assert grad_b.shape == bias.shape
        return grad_b
    def grad_func_inputs(grad):
        grad = grad.reshape(z.shape)
        #对卷积核沿着宽高两个维度进行翻转
        flip_kernel = np.flipud(np.fliplr(np.flip(kernel))).transpose(1,0,2,3)
        N,C,H,W = inputs.shape
        if stride > 1:
            #假设stride为1时forward的卷积图大小
            out_h = (H + 2*padding_h - kernel_size_h)//1 + 1
            out_w = (W + 2*padding_w - kernel_size_w)//1 + 1 
            temp_delta = np.zeros((z.shape[0],z.shape[1],out_h,out_w))
            temp_delta[:,:,::stride,::stride] = grad
            grad = temp_delta
        #delta_2 = np.pad(delta_2,[(0,0), (0,0), (self.kernel_size_h-1, self.kernel_size_h-1), (self.kernel_size_w-1, self.kernel_size_w-1)],'constant')
        grad = im2col(grad,(kernel_size_h,kernel_size_w),1,(kernel_size_h-1,kernel_size_w-1))
        flip_kernel = im2col(flip_kernel,(kernel_size_h,kernel_size_w)).T
        delta_1 = np.dot(grad,flip_kernel).reshape(inputs.shape[0],inputs.shape[2]+2*padding_h,inputs.shape[3]+2*padding_w,C).transpose(0,3,1,2)
        delta_1 = delta_1[:,:,padding_h:delta_1.shape[2]-padding_h,padding_w:delta_1.shape[3]-padding_w]
        assert delta_1.shape == inputs.shape
        return delta_1
    return create_trinary_op_tensor(z,img,kernel_t,bias_t,grad_func_inputs,grad_func_kernel,grad_func_bias)

def maxpool_(img,size,stride):
    inputs = img.values
    assert len(inputs.shape) == 4 and inputs.shape[2] >= size and inputs.shape[3] >= size
    N,C,H,W = inputs.shape
    shape = inputs.shape
    out_h = (H - size)//stride + 1
    out_w = (W - size)//stride + 1 
    tempinputs = im2col(inputs,size,stride) #[N*out_h*out_w ,C * kernel_size_h * kernel_size_w]
    tempinputs = tempinputs.reshape(N*out_h*out_w*C,size * size)
    outputs = np.max(tempinputs,axis = 1,keepdims=True)
    index = np.argmax(tempinputs,axis=1)      #[N,C,out_h*out_w]
    outputs=outputs.reshape(N,out_h,out_w,C).transpose(0,3,1,2)
    def grad_func_inputs(grad):
        #[N*out_h*out_w*C,self.size * self.size]
        N,C,H,W = shape
        out_h = (H - size)//stride + 1
        out_w = (W - size)//stride + 1 
        grad = grad.reshape((N,C,out_h,out_w))
        delta_1 = np.zeros((N*out_h*out_w*C,size*size))
        grad = grad.transpose(0,2,3,1).reshape(-1,1)    #[N,out_h,out_w,C] -> (1,-1)
        delta_1[range(delta_1.shape[0]),index] = grad.reshape(grad.shape[0])
        delta_1 = delta_1.reshape(N*out_h*out_w,-1)
        delta_1 = col2img(delta_1,size,shape,stride)
        return delta_1
    return create_unary_op_tensor(outputs,img,grad_func_inputs)

def softmax__(img):
    s = img.values
    max = np.max(s)
    value = np.exp(s-max) / np.sum(np.exp(s-max),axis = 1,keepdims=True)
    def grad_func_img(grad):
        size = value.shape[1]
        batch_size = value.shape[0]
        array = np.array([])
        for i in range(batch_size):
            array = np.append(array,-np.dot(value[i:i+1,:].T,value[i:i+1,:]) +
                                                    np.identity(size) * value[i:i+1,:])
        array = array.reshape(batch_size,size,size)
        delta1 = np.matmul(np.expand_dims(grad,1),array)
        delta1 = np.squeeze(delta1,1)
        return delta1
    return create_unary_op_tensor(value,img,grad_func_img)

def neg_(tensor1):
    values = -tensor1.values

    def grad_func_ts1(grad):
        return -grad 
    
    return create_unary_op_tensor(values, tensor1, grad_func_ts1)

def exp_(tensor1):
    values = np.exp(tensor1.values)

    def grad_func_ts1(grad):
        return grad * values

    return create_unary_op_tensor(values, tensor1, grad_func_ts1)

def log_(tensor1):
    values = np.log(tensor1.values)

    def grad_func_ts1(grad):
        return grad / tensor1.values
    
    return create_unary_op_tensor(values, tensor1, grad_func_ts1)

def sin_(tensor1):
    values = np.sin(tensor1.values)
    def grad_func_ts1(grad):
        return  grad * np.cos(tensor1.values)
    
    return create_unary_op_tensor(values, tensor1, grad_func_ts1)

def transpose_(tensor1, axes=None):
    values = tensor1.values.transpose(axes)

    if axes is None:
        axes = reversed(range(tensor1.values.ndim))
    axes = list(axes)

    def grad_func_ts1(grad):
        return grad.transpose(np.argsort(axes))

    return create_unary_op_tensor(values, tensor1, grad_func_ts1)

def sum_(tensor1, axis=None, keepdims=False):
    values = tensor1.values.sum(axis=axis, keepdims=keepdims)
    '''
    if axis is not None:
        repeat_num = tensor1.values.shape[axis]
    '''
    def grad_func_ts1(grad):
        '''
        if axis is not None:
            if not keepdims:
                grad = np.expand_dims(grad, axis)
            grad = np.repeat(grad, repeat_num, axis)
        else:
            grad = grad * np.ones_like(tensor1.values)
        return grad 
        '''
        if axis is None or keepdims:
            return grad * np.ones_like(tensor1.values)
        elif type(axis) is int:
            return np.expand_dims(grad, axis=axis)*np.ones_like(tensor1.values)
        else:
            for i in sorted(axis):
                grad = np.expand_dims(grad, axis=i)
            return grad * np.ones_like(tensor1.values)
    return create_unary_op_tensor(values, tensor1, grad_func_ts1)

def clip_(tensor1, low=None, high=None):
    values = tensor1.values.clip(low, high)

    mask = np.ones(tensor1.shape, dtype=bool)
    if low is not None:
        mask &= (tensor1.values >= low)
    if high is not None:
        mask &= (tensor1.values <= high)

    def grad_func_ts1(grad):
        return grad * mask 
    
    return create_unary_op_tensor(values, tensor1, grad_func_ts1)

def getitem_(tensor1, idx):
    values = tensor1.values[idx]

    def grad_func_ts1(grad):
        grads = np.zeros_like(tensor1.values)
        grads[idx] = grad 
        return grads
    
    return create_unary_op_tensor(values, tensor1, grad_func_ts1)

#==========================================================================================
# 在tensor.py 之外的文件中调用时使用的wrapper_function
def exp(obj, requires_grad=False):
    return exp_(convert_to_tensor(obj, requires_grad))

def clip(obj, low=None, high=None, requires_grad=False):
    return (clip_(convert_to_tensor(obj), low, high))

def pow(obj1, obj2, requires_grad=False):
    return pow_(convert_to_tensor(obj1, requires_grad), convert_to_tensor(obj2, requires_grad))

def log(obj1, requires_grad=False):
    return log_(convert_to_tensor(obj1, requires_grad))





