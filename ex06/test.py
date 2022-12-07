from ridge import MyRidge
import numpy as np

x = np.array([
[ -6, -7, -9],
[ 13, -2, 14],
[ -7, 14, -1],
[ -8, -4,
6],
[ -5, -9,
6],
[ 1, -5, 11],
[ 9, -11,
8]])

y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
theta = np.array([[7.01], [3], [10.5], [-6]])
myRidge = MyRidge(theta, 0.01, 1000, 3, True)


# for param in myRidge.get_params_():
#     print(param)

myRidge.set_params_(max_iter=10000, lambda_=3)

# for param in myRidge.get_params_():
#     print(param)


y_hat = myRidge.predict_(x)
print(f"MSE = {MyRidge.mse_(y, y_hat)}")
print(f"loss = {myRidge.loss_(y, y_hat)}")

myRidge.fit_(x, y)
y_hat = myRidge.predict_(x)
print(f"MSE = {MyRidge.mse_(y, y_hat)}")
print(f"loss = {myRidge.loss_(y, y_hat)}")
print(f"loss_elem = {myRidge.loss_elem_(y, y_hat)}")