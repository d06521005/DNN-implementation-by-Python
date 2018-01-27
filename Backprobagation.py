import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

#----------------------------------------
# common function

def cross_entropy_err(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    if t.size == y.size:
        t = t.argmax(axis=1)    
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t.astype(int)] + 1e-7)) / batch_size


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 
    x = x - np.max(x) 
    return np.exp(x) / np.sum(np.exp(x))


#------------------------------------------
# All kinds of function in DNN
class Relu:
    # Relu function with forword & backward
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
        
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx
        
      
class Sigmoid:
    # Sigmoid function with forword & backward
    def __init__(self):
        self.out = None
        
    def forward(self, x):
        out = 1 / (1+np.exp(-x))
        self.out = out
        return out
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * slef.out
        return dx
        
        
class Affine:
    # array dot with forword & backward
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dw = None
        self.db = None
        
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx      
        
        
class SoftmaxWithLoss:
    # Softmax & cross_entropy
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
        
    def forward(self, y, t):
        self.t = t
        self.y = softmax(y)
        self.loss = cross_entropy_err(self.y, self.t)
        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx
        
        
#----------------------------------------------
# DNN model with two layers

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # initialization weight
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
        # generate each layers
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y==t) / float(x.shape[0])
        return accuracy
    
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        #backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        # setting
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        return grads


#----------------------------------------------
# Get MNIST data

def _shuffle(x, y):
    # shuffle the order
    random = np.arange(x.shape[0])
    np.random.shuffle(random)
    return x[random], y[random]
    

def splid_valid_set(x, y, val_percen=0.2):
    who_size = x.shape[0]
    val_size = int(who_size * val_percen)
    x_all, y_all = _shuffle(x, y)
    
    x_train, y_train = x_all[0:val_size], y_all[0:val_size]
    x_valid, y_valid = x_all[val_size:], y_all[val_size:]
    return x_train, y_train, x_valid, y_valid

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
    
    
def get_onehot(y):
    length = np.unique(y).shape[0]
    OneHot = np.zeros((y.shape[0], length))
    OneHot[np.arange(y.shape[0]), y.astype(int)] = 1
    return OneHot


from sklearn.datasets import fetch_mldata
from PIL import Image
mnist = fetch_mldata('MNIST original')

X_data = mnist.data
Y_data = mnist.target
Answer = get_onehot(Y_data) 
x_train, y_train, x_test, y_test = splid_valid_set(X_data, Answer)


#---------------------------------------
# Train data
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = y_train[batch_mask]
    
    # 勾配
    grad = network.gradient(x_batch, t_batch)
    
    # 更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, y_train)
        test_acc = network.accuracy(x_test, y_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)
        
