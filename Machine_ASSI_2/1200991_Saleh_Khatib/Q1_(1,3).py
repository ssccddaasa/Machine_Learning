import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

pd.options.display.max_rows = 9999

# read dataset from the file and print the info of it.
data_set = pd.read_csv("data_reg.csv")

print(data_set.info())


# Q1-1 read split and draw.
data_set_train = data_set.iloc[:120]
data_set_vaild = data_set.iloc[120:160]
data_set_test = data_set.iloc[160:]




fig = plt.figure()

ax = fig.add_subplot(projection='3d')

ax.scatter(data_set_train["x1"],data_set_train["x2"],data_set_train["y"],c="red",marker='o',label="Train set")
ax.scatter(data_set_vaild["x1"],data_set_vaild["x2"],data_set_vaild["y"],c="blue",marker='o',label="Vaild set")
ax.scatter(data_set_test["x1"],data_set_test["x2"],data_set_test["y"],c="green",marker='o',label="Test set")

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title('3D Scatter Plot')

plt.legend()

# Show plot
plt.show()




x_train = data_set_train[["x1","x2"]]
y_train = data_set_train["y"]

x_valid = data_set_vaild[["x1","x2"]]
y_valid = data_set_vaild["y"]


# Q1-3 best parm.

# make the parms
rid_parm = [0.001, 0.005, 0.01, 0.1, 10]

degree = 8

valid_error = []

poly = PolynomialFeatures(degree=degree)

x_train_p = poly.fit_transform(x_train)
x_valid_p = poly.fit_transform(x_valid)

# loop to make module and the VE.
for alpha in rid_parm:

    model = Ridge(alpha=alpha)
    model.fit(x_train_p,y_train)


    y_prd_val = model.predict(x_valid_p)

    error = mean_squared_error(y_valid,y_prd_val)
    valid_error.append(error)


# plot the result
plt.figure()
plt.plot(rid_parm,valid_error,marker="o")
plt.xlabel('Regularization Parameter')
plt.ylabel('Validation Error (MSE)')
plt.title('Validation Error vs Regularization Parameter (Ridge Regression)')
plt.xscale('log')
plt.grid(True)
plt.show()

#best parm is 0.01.