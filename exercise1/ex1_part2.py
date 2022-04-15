import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



# load the data
data_path = r"/Users/jiakunxin/githubRepos/machineLearningExercise/exercise1/ex1data2.txt"
data = pd.read_csv(data_path, sep=",", header=None, index_col=False, names=['size', 'bedroom_cnt', 'price'])

# feature normalization
size_mean = data['size'].mean()
size_std = data['size'].std()
bedrm_mean = data['bedroom_cnt'].mean()
bedrm_std = data['bedroom_cnt'].std()

data['size_normalized'] = (data['size'] - size_mean) / size_std
data['bedroom_cnt_normalized'] = (data['bedroom_cnt'] - bedrm_mean) / bedrm_std

# implementing gradient descent
m = len(data)  # training data size
n_iterations = 5000
alphas = [0.3, 0.1, 0.03, 0.01]
x = np.column_stack([np.ones(m), data.size_normalized, data.bedroom_cnt_normalized])
y = np.array(data.price)
y_pred = np.empty(m)
theta_alpha = []
error_alpha = []

for alpha in alphas:
    errors = []
    theta = np.zeros(3)
    for i in range(n_iterations):
        y_pred = np.matmul(x, theta)
        error = np.dot(y_pred-y, y_pred-y)/(2*m)
        errors.append(error)
        theta = theta - 1/m * alpha * np.matmul(x.T, y_pred-y)
    error_alpha.append(errors)
    theta_alpha.append(theta)

# visualizing J(theta)
# 可以看到收敛速度是不一样的
plt.plot(np.arange(n_iterations), error_alpha[0], color='red', label='alpha: 0.3')
plt.plot(np.arange(n_iterations), error_alpha[1], color='blue', label='alpha: 0.1')
plt.plot(np.arange(n_iterations), error_alpha[2], color='green', label='alpha: 0.03')
plt.plot(np.arange(n_iterations), error_alpha[3], color='yellow', label='alpha: 0.01')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title('Error vs. Training Epochs')
plt.legend()
plt.show()

for i in range(len(alphas)):
    print(f"Parameters: {theta_alpha[i]}")

# 用来验证gradient descent没写错
model = LinearRegression()
model.fit(np.array(x[:,1:]), y.reshape(-1,1))
print(model.coef_)
print(model.intercept_)