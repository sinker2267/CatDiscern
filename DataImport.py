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
print(train_data_std[:9,:9])
