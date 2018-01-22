import numpy as np
import matplotlib.pyplot as plt

# Common Function
def cross_entropy_err(y, t):
    # cross_entropy
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))
    
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c) # prevent exp(a) too large
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
 
def sigmoid(x):
    res = 1 / (1.0 + np.exp(-x))
    return np.clip(res, 1e-8, 1-(1e-8))

def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
    
# Two Layers NN 
class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {} 
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y
        

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_err(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads
      
      
        
# Train MNIST data
  
from sklearn.datasets import fetch_mldata
from PIL import Image
mnist = fetch_mldata('MNIST original')

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

X_data = mnist.data
Y_data = mnist.target
print(X_data.shape, Y_data.shape)

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
    
    grad = network.gradient(x_batch, t_batch)
    
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, y_train)
        test_acc = network.accuracy(x_test, y_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
