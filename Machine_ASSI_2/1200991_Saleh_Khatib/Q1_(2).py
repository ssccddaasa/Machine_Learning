import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

pd.options.display.max_rows = 9999


#Q1-2 best degree.
# read dataset from the file and print the info of it.
data_set = pd.read_csv("data_reg.csv")

data_set_train = data_set.iloc[:120]
data_set_vaild = data_set.iloc[120:160]
data_set_test = data_set.iloc[160:]

# make the sets
x_train = data_set_train[["x1","x2"]]
y_train = data_set_train["y"]

x_valid = data_set_vaild[["x1","x2"]]
y_valid = data_set_vaild["y"]


degrees = np.arange(1,11)

valid_error = []
fig = plt.figure(figsize=(15, 10))


for degree in degrees:

    # make the features  Polynomial Features
    poly = PolynomialFeatures(degree=degree, include_bias=False)

    x_train_p = poly.fit_transform(x_train)
    x_valid_p = poly.fit_transform(x_valid)

# make the module
    model = LinearRegression()
    model.fit(x_train_p,y_train)
# make the 3d axe
    my_3dax = fig.add_subplot(3,4,degree,projection="3d")
    my_3dax.scatter(data_set_train["x1"],data_set_train["x2"],data_set_train["y"],c="red",marker='o',label="Train set")


# make the data ready to plot for surface of the learned function alongside with the training examples.
    x1r = np.linspace(min(data_set_train["x1"]), max(data_set_train["x1"]), 100)
    x2r = np.linspace(min(data_set_train["x2"]), max(data_set_train["x2"]), 100)
    x1, x2 = np.meshgrid(x1r,x2r)

    xt_mesh = np.c_[x1.ravel(), x2.ravel()]
    xt_mesh_p = poly.transform(xt_mesh)
    y_mesh = model.predict(xt_mesh_p)
    y_mesh = y_mesh.reshape(x1.shape)
    

    my_3dax.plot_surface(x1,x2,y_mesh,alpha=0.5, cmap="viridis", label= f'degree {degree}')

    my_3dax.set_xlabel('X1')
    my_3dax.set_ylabel('X2')
    my_3dax.set_zlabel('Y')

    my_3dax.set_title(f'Polynomial Degree {degree}')

    y_prd_val = model.predict(x_valid_p)
    error = mean_squared_error(y_valid,y_prd_val)
    valid_error.append(error)


plt.tight_layout()
plt.show()

# plot the results.
plt.figure()
plt.plot(degrees,valid_error)
plt.xlabel('Polynomial Degree')
plt.ylabel('Validation Error (MSE)')
plt.title('Validation Error vs Polynomial Degree')
plt.grid(True)
plt.show()

