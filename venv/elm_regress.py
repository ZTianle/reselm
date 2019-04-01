import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from ELM_myself import HiddenLayer

# from DELM_myself import HiddenLayer
data = pd.read_csv('r_test.csv')
x = data.x.values.reshape(-1, 1)
y = data.y.values.reshape(-1, 1)

my_ELM = HiddenLayer(x, 5)
my_ELM.regressor_train(y)
x_test = np.linspace(0.9, 5.02, 100).reshape(-1, 1)
y_test = my_EML.regressor_test(x_test)
plt.plot(x_test, y_test)
plt.scatter(x, y)
plt.title('ELM_regress')
plt.xlabel('x')
plt.ylabel('y')
plt.show()#