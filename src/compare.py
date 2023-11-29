import numpy as np
from sklearn.linear_model import LinearRegression
from multiple_linear_regression import MultipleLinearRegression, mean_squared_error
from sklearn import datasets

gen = np.random.default_rng()
X_values = gen.normal(size=(5, 3))

# Target vector (y)
y_values = np.array([5, 7, 9, 11, 13])

X = X_values  # Extracting the values from the features DataFrame
y = y_values   # Extracting the values from the target Series

# Train and predict with your implementation
model = MultipleLinearRegression()
model.train(X, y)
predictions = model.predict(X)

# Compare with sklearn's implementation
sklearn_model = LinearRegression()
sklearn_model.fit(X, y)
sklearn_predictions = sklearn_model.predict(X)

# Compare coefficients and predictions
print("Your model coefficients:", (model._intercept, *model._slope))
print("Sklearn model coefficients:", [sklearn_model.intercept_, *sklearn_model.coef_])

print("Mean squared error of your model:", mean_squared_error(y, predictions))
print("Mean squared error of sklearn model:", mean_squared_error(y, sklearn_predictions))


#############################################################################################################################
###                                                 ANOTHER TESTING                                                       ###
#############################################################################################################################
diabetes = datasets.load_diabetes()
model = MultipleLinearRegression()
observations = diabetes.data

model.train(diabetes.data, diabetes.target)

predictions = model.predict(diabetes.data)

# Compare with sklearn's implementation
sklearn_model = LinearRegression()
sklearn_model.fit(diabetes.data, diabetes.target)
sklearn_predictions = sklearn_model.predict(diabetes.data)


print("Real: ", diabetes.target[:5])
print("Pred: ", predictions[:5])


# Compare coefficients and predictions
print("Your model coefficients:", (model._intercept, *model._slope))
print("\nSklearn model coefficients:", [sklearn_model.intercept_, *sklearn_model.coef_])

print("MSE: {}".format(mean_squared_error(diabetes.target, predictions)))
print("MSE: {}".format(mean_squared_error(diabetes.target, sklearn_predictions)))
