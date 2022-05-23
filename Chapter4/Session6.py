
def activation_function(x):
   return 1/(1+np.exp(-x)) # sigmoid function
#    return np.where(x<=0,0,1) # step function
#    return np.where(x<=0,0,x) # ReLu function
#    return np.where(x <= 0, 0.01*x, x)  # leakyReLu function


"""
Session 4.6.1~4.6.2
"""
import numpy as np
import matplotlib.pyplot as plt

X = np.arange(-1,1,0.2)
Y = np.arange(-1,1,0.2)
Z = np.zeros((10,10))

w_x = 2.5
w_y = 3.0

bias = 0.1


for i in range(10):
    for j in range(10):
        u = X[i]*w_x + Y[j]*w_y + bias
        y = activation_function(u)

        Z[j][i] = y

# plt.imshow(Z,'gray',vmin=0.0,vmax=1.0)
# plt.colorbar()
# plt.show()


"""
Session 4.6.3~4.6.6
"""

def middle_layer(x,w,b): # using sigmoid for activation
    u=np.dot(x,w)+b
    return activation_function(u)

def output_layer(x,w,b): # using identity function for regression
    return np.dot(x,w)+b

w_im = np.array([[4,4],
                 [4,4]]) # input : 2 , middle : 2 therefore 2x2 matrix
w_mo = np.array([[1],
                 [-1]]) # middle : 2 , output : 1 therefore 2x1 matrix
b_im = np.array([3,-3]) # input : 1x2, middle : 2x2 -> 1x2 vector
b_mo = np.array([0.1]) # middle : 1x2, output : 1x1 therefore 1x1 vector

Z = np.zeros((10,10))

for i in range(10):
    for j in range(10):
        inp = np.array([X[i],Y[j]])
        mid = middle_layer(inp,w_im,b_im)
        out = output_layer(mid,w_mo,b_mo)

        Z[j][i] = out

# plt.imshow(Z,"gray",vmin=0.0,vmax=1.0)
# plt.colorbar()
# plt.show()



"""
Session 4.6.7~4.6.6
"""

def sigmoid_function(x):
   return 1/(1+np.exp(-x)) # sigmoid function
def softmax_function(x):
    return np.exp(x)/np.sum(np.exp(x))
def output_layer(x,w,b):
    u=np.dot(x,w)+b
    return softmax_function(u)

X = np.arange(-1,1,0.1)
Y = np.arange(-1,1,0.1)

x,y = np.meshgrid(X,Y)

w_im = np.array([[1,2],[2,3]])
w_mo = np.array([[-1,1],[1,-1]])
b_im = np.array([0.3,-0.3])
b_mo = np.array([0.4,0.1])

input_ = np.vstack((x.flatten(),y.flatten()))#np.vstack((X,Y))
mid = sigmoid_function(np.dot(input_.transpose(),w_im)+b_im)
out = softmax_function(np.dot(mid,w_mo)+b_mo)

bools = out[:,0]>out[:,1]

class_1 = input_.transpose()[bools]
class_2 = input_.transpose()[np.logical_not(bools)]

plt.scatter(class_1[:,0],class_1[:,1],marker="+")
plt.scatter(class_2[:,0],class_2[:,1],marker="o")
plt.show()

#
# for i in range(20):
#     if out[i,0] > out[i,1] :
#
#     print(out[i,:])
# for i in range(20):
#     for j in range(20):
#
#         inp = np.array(X[i],Y[j])
#         mid = middle_layer(inp,w_im,b_im)
#         out = output_layer(mid,w_mo,b_mo)
#
