#导入数据
import h5py
import matplotlib.pyplot as plt
import numpy as np
train_dataset = h5py.File('datasets/train_catvnoncat.h5','r')
test_dataset = h5py.File('datasets/test_catvnoncat.h5','r')
#取出训练集 测试集
train_data_org = train_dataset['train_set_x'][:]
train_labels_org = train_dataset['train_set_y'][:]
test_data_org = test_dataset['test_set_x'][:]
test_labels_org = test_dataset['test_set_y'][:]
#数据维度处理
m_train = train_data_org.shape[0]
m_test = test_data_org.shape[0]
train_data_trans = train_data_org.reshape(m_train,-1).T
test_data_trans = test_data_org.reshape(m_test,-1).T

#标签处理
train_labels_trans = train_labels_org[np.newaxis,:]
test_labels_trans = test_labels_org[np.newaxis,:]

#标准化数据 归一化处理
# print(train_data_trans[:9,:9])
train_data_std = train_data_trans / 255 # data - min /(max - min) ; min:0;max:255
test_data_std = test_data_trans / 255

#定义sigmoid函数
def sigmoid(z):
    a = 1 / (1 + np.exp(-z))
    return a

#初始化参数
n_dim = train_data_std.shape[0]
w = np.zeros([n_dim,1])
b = 0

#定义前向传播函数，代价函数以及梯度下降
def propagate(w, b, X, y):
    #1.前向传播函数
    A = sigmoid(np.dot(w.T, X) + b)

    #2.代价函数
    m = X.shape[1]
    J = -1 / m * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))

    #3.梯度下降
    dw = 1 / m * np.dot(X, (A - y).T)
    db = 1 / m * np.sum(A - y)

    grands = {'dw':dw, 'db':db}
    return grands, J


#优化部分
#n_iters 迭代次数
def optimize(w,b,X,y,alpha,n_iters):
    costs = []
    for i in range(n_iters - 1):
     grands,J = propagate(w,b,X,y)
     dw = grands['dw']
     db = grands['db']

     w = w - alpha * dw
     b = b - alpha * db

     if i % 100 == 0:
         costs.append(J)
         print("n_iters is ",i,' cost is ',J)
    grands = {'dw': dw, 'db': db}
    params = {'w':w, 'b':b}
    return grands, params
