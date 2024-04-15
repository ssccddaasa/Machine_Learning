import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro 


pd.options.display.max_rows = 9999

# read dataset from the file and print the info of it.
data_set = pd.read_csv("cars.csv")

print(data_set.info())

#nds = data_set.isnull()


# here we get median and mode to fill.
print(data_set[data_set.columns[3]].median)

# 130

print(data_set[data_set.columns[7]].mode)

# USA


data_set["horsepower"].fillna(130, inplace=True)
data_set["origin"].fillna("USA", inplace=True)

print(data_set.info())
#print(data_set)





# here we plot boxplot for mpg.
data_set.boxplot(by="origin",column=['mpg',], grid= False)
# here we plot histogram.
data_set.hist(column=["mpg","horsepower","acceleration"], bins=50)
# here we plot scatter plot.
data_set.plot.scatter(x = "horsepower", y = "mpg")
plt.show()
# shaprio test.
print("for mpg: ",shapiro(data_set["mpg"]))
print("for horsepowr: ",shapiro(data_set["horsepower"]))
print("for acceleration: ",shapiro(data_set["acceleration"]))








# linear regression
X = data_set["horsepower"].values.reshape(-1,1)
Y = data_set["mpg"].values.reshape(-1,1)

# add ones.
X_b = np.c_[np.ones((len(X), 1)),X]

# get w
sol = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y)
print("\n\nthe W(lin) are: ",sol)
# plot result
plt.scatter(X, Y, label='Data Points')
plt.plot(X, X_b.dot(sol), color='red', label='Learned Line')

plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.title('Linear Regression: MPG vs Horsepower')
plt.legend()
plt.show()

# non-linear
# get z from polynomial function 
z = np.c_[X, X**2]
# add ones.
z_b = np.c_[np.ones((len(z), 1)),z]

# get w
sol2 = np.linalg.inv(z_b.T.dot(z_b)).dot(z_b.T).dot(Y)
# plot result
plt.scatter(X, Y, label='Data Points')

x_val = np.linspace(np.min(X), np.max(X), 100)
Y_val = sol2[0][0] + sol2[1][0]*x_val + sol2[2][0]* (x_val**2)
print("\n\nthe W(nonlin) are: ",sol2)
plt.plot(x_val, Y_val, color='green', label='Learned Line')
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.title('Non-Linear Regression: MPG vs Horsepower')
plt.legend()
plt.show()

# gradient

# scale X and make it near normal dis.
X = (X - np.mean(X)) / np.std(X)
# add ones.
X_b = np.c_[np.ones((len(X), 1)),X]
A = 0.05
np.random.seed(433)
w = np.random.rand(2,1)


x_len = len(X)

for i in range(1000):

    Y_P = np.dot(X_b, w)
    er = Y_P - Y
    gred = (1/x_len)* np.dot(X_b.T, er)
    w -= A*gred



Y_P2 = w[1][0] * X + w[0][0]
print("\n\nthe W(grd) are: ",w)
plt.scatter(X, Y, label='Data Points')

plt.plot(X,Y_P2, color='red', label='Learned Line')

plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.title('Gradient descent algorithm: MPG vs Horsepower')
plt.legend()



plt.show()
