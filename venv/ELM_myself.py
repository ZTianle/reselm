import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.datasets import load_iris  # 数据集
from sklearn.model_selection import train_test_split  # 数据集的分割函数
from sklearn.preprocessing import StandardScaler  # 数据预处理
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn import metrics


class HiddenLayer:
    def __init__(self, x, num):

        self.data_x = np.atleast_2d(x)  # 判断输入训练集是否大于等于二维; 把x_train()取下来
        self.num_data = len(self.data_x)  # 训练数据个数
        self.num_feature = self.data_x.shape[1];  # shape[] 读取矩阵的长度，比如shape[0]就是读取矩阵第一维度的长度 (120行，4列，所以shape[0]==120,shape[1]==4)
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

        h = self.sigmoid(np.dot(x, self.w) + self.b)
        self.H_ = np.linalg.pinv(h)
        # print(self.H_.shape)

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def regressor_train(self, T):
        T = T.reshape(-1, 1)
        self.beta = np.dot(self.H_, T)
        return self.beta

    def classifisor_train(self, T):
        en_one = OneHotEncoder()
        T = en_one.fit_transform(T.reshape(-1, 1)).toarray()  # 独热编码之后一定要用toarray()转换成正常的数组
        # T = np.asarray(T)
        print(self.H_.shape)
        print(T.shape)
        self.beta = np.dot(self.H_, T)
        print(self.beta.shape)
        return self.beta

    def regressor_test(self, test_x):
        b_row = test_x.shape[0]
        h = self.sigmoid(np.dot(test_x, self.w) + self.b[:b_row, :])
        result = np.dot(h, self.beta)
        return result

    def classifisor_test(self, test_x):
        b_row = test_x.shape[0]
        h = self.sigmoid(np.dot(test_x, self.w) + self.b[:b_row, :])
        result = np.dot(h, self.beta)
        result = [item.tolist().index(max(item.tolist())) for item in result]
        return result


stdsc = StandardScaler()
iris = load_iris()
x, y = stdsc.fit_transform(iris.data), iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

a = HiddenLayer(x_train, 20)
a.classifisor_train(y_train)
result = a.classifisor_test(x_test)

print(result)
print(metrics.accuracy_score(y_test, result))
