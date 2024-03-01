import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

pd.options.display.max_rows = 9999

# Q2 -1
# read dataset from the file and print the info of it.
data_set_test = pd.read_csv("test_cls.csv")

data_set_train = pd.read_csv("train_cls.csv")

print(data_set_train.info())

# make the sets
x_train = data_set_train[["x1","x2"]]
y_train = data_set_train["class"]

x_test = data_set_test[["x1","x2"]]
y_test = data_set_test["class"]


# encode the class
lab = LabelEncoder()
y1 = lab.fit_transform(y_train)
y2 = lab.fit_transform(y_test)

# make the module
model = LogisticRegression()
model.fit(x_train,y1)

y_prd = model.predict(x_test)

# print Classification Reports
print("Classification Report test:")
print(classification_report(y2, y_prd))
y_prd1 = model.predict(x_train)
print("Classification Report train:")
print(classification_report(y1, y_prd1))


# make the data ready to plot.
plt.figure()
plt.scatter(data_set_train["x1"],data_set_train["x2"], c= data_set_train["class"],marker='o')

plt.title('Logistic Regression - Decision Boundary')
plt.xlabel('x1')
plt.ylabel('x2')

x1_lin1, x1_lin2 = data_set_train["x1"].min() - 1, data_set_train["x1"].max() + 1
x2_lin1, x2_lin2 = data_set_train["x2"].min() - 1, data_set_train["x2"].max() + 1

x1, x2 = np.meshgrid(np.linspace(x1_lin1, x1_lin2, 100), np.linspace(x2_lin1, x2_lin2, 100))
Z = model.predict(np.c_[x1.ravel(), x2.ravel()])
Z = Z.reshape(x1.shape)

plt.contourf(x1, x2, Z, alpha=0.4, cmap='viridis')




plt.legend()
plt.show()