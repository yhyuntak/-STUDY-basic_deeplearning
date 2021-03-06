import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris_data = datasets.load_iris()
input_data = iris_data.data
correct = iris_data.target
n_data = len(correct)

# 데이터 standardization
ave_input = np.average(input_data,axis=0)
std_input = np.std(input_data,axis=0)
input_data = (input_data-ave_input)/std_input

# 원핫인코딩
correct_data = np.zeros((n_data,3))
for i in range(n_data):
    correct_data[i,correct[i]]=1.0

#사실 여기에 sklearn에서의 기법들을 이용해서 하면 데이터 전처리가 전부 되긴함.


index = np.arange(n_data)
index_train = index[index%2==0]
index_test = index[index%2 != 0]
input_train = input_data[index_train,:]
correct_train = correct_data[index_train,:]
input_test = input_data[index_test,:]
correct_test = correct_data[index_test,:]

n_train = input_train.shape[0]
n_test = input_test.shape[0]

n_in =4
n_mid = 25
n_out = 3

wb_width = 0.1
eta = 0.01
epoch = 100
batch_size = 8
interval = 10

class BaseLayer:
    def __init__(self,n_upper,n):
        self.w = wb_width*np.random.randn(n_upper,n)
        self.b = wb_width * np.random.randn(n)

    def update(self,eta):
        self.w -= eta*self.grad_w
        self.b -= eta*self.grad_b

class MiddleLayer(BaseLayer):
    def forward(self,x):
        self.x = x
        self.u = np.dot(x,self.w)+self.b
        self.y = np.where(self.u <=0,0,self.u)
    def backward(self,grad_y):
        delta = grad_y * np.where(self.u<=0,0,1)
        self.grad_w = np.dot(self.x.T,delta)
        self.grad_b = np.sum(delta,axis=0)
        self.grad_x =np.dot(delta,self.w.T)

class OutputLayer(BaseLayer):
    def forward(self,x):
        self.x=x
        u = np.dot(x,self.w)+self.b
        self.y = np.exp(u)/np.sum(np.exp(u),axis=1,keepdims=True)

    def backward(self,t):
        delta = self.y -t
        self.grad_w = np.dot(self.x.T,delta)
        self.grad_b = np.sum(delta,axis=0)
        self.grad_x=np.dot(delta,self.w.T)

middle_layer1 = MiddleLayer(n_in,n_mid)
middle_layer2 = MiddleLayer(n_mid,n_mid)
output_layer = OutputLayer(n_mid,n_out)

def forward_propagation(x):
    middle_layer1.forward(x)
    middle_layer2.forward(middle_layer1.y)
    output_layer.forward(middle_layer2.y)

def backpropagation(t):
    output_layer.backward(t)
    middle_layer2.backward(output_layer.grad_x)
    middle_layer1.backward(middle_layer2.grad_x)

def update_wb():
    middle_layer1.update(eta)
    middle_layer2.update(eta)
    output_layer.update(eta)

def get_error(t,batch_size):
    return -np.sum(t*np.log(output_layer.y+1e-7))/batch_size

train_error_x = []
train_error_y = []
test_error_x =[]
test_error_y = []

n_batch = n_train // batch_size
for i in range(epoch):
    forward_propagation(input_train)
    error_train = get_error(correct_train,n_train)
    forward_propagation(input_test)
    error_test = get_error(correct_test,n_test)

    test_error_x.append(i)
    test_error_y.append(error_test)
    train_error_x.append(i)
    train_error_y.append(error_train)

    if i % interval == 0:
        print("Epoch:"+str(i)+"/"+str(epoch),
              "Error_train:"+str(error_train),
              "Error_test:"+str(error_test))

    index_random = np.arange(n_train)
    np.random.shuffle(index_random)

    for j in range(n_batch):
        mb_index = index_random[j*batch_size : (j+1)*batch_size]
        x=input_train[mb_index,:]
        t= correct_train[mb_index,:]

        forward_propagation(x)
        backpropagation(t)
        update_wb()

plt.plot(train_error_x,train_error_y,label="Train")
plt.plot(test_error_x,test_error_y,label="Test")
plt.legend()

plt.xlabel("Epochs")
plt.ylabel("Error")

plt.show()

forward_propagation(input_train)
count_train = np.sum(np.argmax(output_layer.y,axis=1)==np.argmax(correct_test,axis=1))
print("훈련 정확도 :",str(count_train/n_train*100)+"%",
      "테스트 정확도 :",str(count_test/n_test*100)+"%")




