import numpy as np
from my_logistic_regression import MyLogisticRegression as mylogr

theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
# Example 1:
model1 = mylogr(theta, lambda_=5.0)
model1.penality
# Output
# ’l2’
model1.lambda_
# Output
# 5.0
# Example 2:
model2 = mylogr(theta, penality=None)
model2.penality
# Output
# None
model2.lambda_
# Output
# 0.0
# Example 3:
model3 = mylogr(theta, penality=None, lambda_=2.0)
model3.penality
# Output
# None
model3.lambda_
# Output
# 0.0