from src.multiple_linear_regression import MultipleLinearRegression

from sklearn.datasets import load_diabetes
import numpy as np

data = load_diabetes()

X = data['data']
Y = data['target']

model = MultipleLinearRegression(n_dims=X.shape[1])
model.train(X, Y)

predictions = model.predict(X)

mse = np.square(Y - predictions).mean()

print(f'Mean Squared Error : {mse}')
