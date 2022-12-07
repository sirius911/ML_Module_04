import numpy as np
from polynomial_model_extended import add_polynomial_features

x = np.arange(1,11).reshape(5, 2)
print(x)
print("*** power=0 ***")
print(add_polynomial_features(x, 0))
print("*** power=1 ***")
print(add_polynomial_features(x, 1))

# Example 1:
print("*** power=3 ***")
print(add_polynomial_features(x, 3))

# Example 2:
print("*** power=4 ***")
print(add_polynomial_features(x, 4))
