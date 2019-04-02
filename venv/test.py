import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from elm_confidence import HiddenLayer
import math
#准备数据 y=sin（x）
x = np.arange(0, 2*np.pi, 2*np.pi/100)
y = np.sin(x)
x = x.reshape(100, 1)
y = y.reshape(100, 1)
y_real = y + np.random.randn(100, 1)/5
plt.plot(x, y)
plt.scatter(x, y_real)
plt.show()

x_train = x[0:80, -1]
x_test = x[80:, -1]
x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)

y_train = x[0:80, -1]
y_test = x[80:, -1]
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

my_ELM = HiddenLayer(x_train, 5, 0.0000001)
my_ELM.regressor_train(y_train)
y_test_ELM, y_test_ELM_c = my_ELM.regressor_test(x_test)
