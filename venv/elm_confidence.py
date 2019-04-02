import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.datasets import load_iris  # 数据集
from sklearn.model_selection import train_test_split  # 数据集的分割函数
from sklearn.preprocessing import StandardScaler  # 数据预处理
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn import metrics

class HiddenLayer:
    def __init__(self, x, num, c):
        self.c=c
        self.data_x = np.atleast_2d(x)  # 判断输入训练集是否大于等于二维; 把x_train()取下来
        self.num_data = len(self.data_x)  # 训练数据个数
        self.num_feature = self.data_x.shape[
            1];  # shape[] 读取矩阵的长度，比如shape[0]就是读取矩阵第一维度的长度 (120行，4列，所以shape[0]==120,shape[1]==4)
        self.num_hidden = num;  # 隐藏层节点个数

        # 随机生产权重（从-1，到1，生成（num_feature行,num_hidden列））
        self.w = np.random.uniform(-1, 1, (self.num_feature, self.num_hidden))

        # 随机生成偏置，一个隐藏层节点对应一个偏置
        for i in range(self.num_hidden):
            b = np.random.uniform(-0.6, 0.6, (1, self.num_hidden))
            self.first_b = b

        # 生成偏置矩阵，以隐藏层节点个数4为行，样本数120为列
        for i in range(self.num_data - 1):
            b = np.row_stack((b, self.first_b))  # row_stack 以叠加行的方式填充数组
        self.b = b

        self.h = self.sigmoid(np.dot(x, self.w) + self.b)
        self.H_ = np.linalg.pinv(self.h)

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def regressor_train(self, T):
       # T = T.reshape(-1, 1)
        self.beta1 = np.dot(self.H_, T)
        self.train_result = np.dot(self.h, self.beta1)
        self.beta2 = 2*self.c*np.linalg.pinv((np.dot(np.dot(T-self.train_result, self.h)), (T-self.train_result)))
        self.T = T
        return self.beta1, self.beta2, self.T

    def regressor_test(self, test_x):
        b_row = test_x.shape[0]
        h = self.sigmoid(np.dot(test_x, self.w) + self.b[:b_row, :])
        result_elm = np.dot(h, self.beta1)
        print(h.shape, self.beta1.shape, self.beta2.shape, result_elm.shape)
        result_elm_c = np.dot(np.dot(h, self.beta2), result_elm)+np.dot(1-np.dot(h, self.beta2), self.T)
        return result_elm, result_elm_c
