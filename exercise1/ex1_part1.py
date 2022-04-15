import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# load the data
data_path = r"/Users/jiakunxin/githubRepos/machineLearningExercise/exercise1/ex1data1.txt"
data = pd.read_csv(data_path, sep=",", header=None, index_col=False, names=['population', 'profit'])

# visualize training data
plt.scatter(x=data.population, y=data.profit, c='red', marker='.')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.title('Scatter Plot of Training Data')
plt.show()

# implementing gradient descent
m = len(data)        # training data size
n_iterations = 5000
alpha = 0.01
theta = np.zeros(2)
x = np.column_stack([np.ones(m), data.population])
y = np.array(data.profit)
y_pred = np.empty(m)
errors = []

for i in range(n_iterations):
    y_pred = np.matmul(x, theta)
    error = np.dot(y_pred-y, y_pred-y)/(2*m)
    errors.append(error)
    theta = theta - 1/m * alpha * np.matmul(x.T, y_pred-y)

x_new1 = np.array([1, 3.5])
x_new2 = np.array([1, 7])
y_predict1 = np.dot(x_new1, theta)
y_predict2 = np.dot(x_new2, theta)
print(f"Parameters: {theta}", f"Predicted result1: {y_predict1}", f"Predicted result2: {y_predict2}", sep='\n')

# plot fitted line
plt.scatter(x=data.population, y=data.profit, c='red', marker='.', label='Training data')
line_x = np.linspace(5, 25, 100)
model_x = np.column_stack([np.ones(100), line_x])
line_y = np.dot(model_x, theta)
plt.plot(line_x, line_y, color='blue', label='Fitted line')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.title('Predicted Profit vs. Population')
plt.legend()
plt.show()

# visualizing J(theta)
plt.plot(np.arange(n_iterations), errors, color='red')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title('Error vs. Training Epochs')
plt.show()

# 用来验证gradient descent没写错
model = LinearRegression()
model.fit(np.array(data.population).reshape(-1,1), np.array(data.profit).reshape(-1,1))
print(model.coef_)
print(model.intercept_)


