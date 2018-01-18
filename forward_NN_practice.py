from sklearn.datasets import fetch_mldata
from PIL import Image

# download the mnist data
mnist = fetch_mldata('MNIST original')


def _shuffle(x, y):
    # shuffle the data order randomly
    random = np.arange(x.shape[0])
    np.random.shuffle(random)
    return x[random], y[random]
    

def splid_valid_set(x, y, val_percen=0.2):
    # split the data into train/validation set
    who_size = x.shape[0]
    val_size = int(who_size * val_percen)
    x_all, y_all = _shuffle(x, y) # shuffle the data
    
    x_train, y_train = x_all[0:val_size], y_all[0:val_size]
    x_valid, y_valid = x_all[val_size:], y_all[val_size:]
    return x_train, y_train, x_valid, y_valid


def img_show(img):
    # print one image  
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
    
    
def get_onehot(y):
    # transform label to onehot code
    length = np.unique(y).shape[0]
    OneHot = np.zeros((y.shape[0], length))
    OneHot[np.arange(y.shape[0]), y.astype(int)] = 1
    return OneHot

import pickle
def init_network():
    # get pretrain weight from .pkl file (supplied by book)
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    # foreware process in NN (two layers)
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    return y   
#np.argwhere(Y_data==5.)  

#========================================
# read the data
X_data = mnist.data
Y_data = mnist.target
print(X_data.shape, Y_data.shape)

# split the data
x_train, y_train, x_test, y_test = splid_valid_set(X_data, Y_data)

#get the pretrain weight
network = init_network()

# Run forward NN
accuracy_cnt = 0
for i in range(x_train.shape[0]):
    y = predict(network, x_train)
    p = np.argmax(y)
    if p == y_train[i]:
        accuracy_cnt += 1
print("Accuracy:" + str(float(accuracy_cnt) / x_train.shape[0]))
