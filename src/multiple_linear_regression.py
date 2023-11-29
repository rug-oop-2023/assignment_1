import numpy as np

class MultipleLinearRegression:
    def __init__(self):
        self._features = 0
        self._intercept = None
        self._slope = None

    def train(self, observations: np.ndarray, target: np.ndarray) -> None:
        '''
        observations is a 2d numpy array of length n rows * p columns
        target is a 1d numpy array of length n
        '''

        self._features = np.shape(observations)[1]

        ones = np.ones((np.shape(observations)[0], 1))
        X = np.hstack((ones, observations))

        X_transpose = X.transpose()
        X_inverse = np.linalg.inv(X_transpose.dot(X))
        weights = X_inverse.dot(X_transpose).dot(target)

        self._intercept = weights.item(0)
        self._slope = np.delete(weights, 0)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        if np.shape(observations)[1] != self._features:
            raise Exception("Wrong number of feature columns")
        return self._intercept + observations.dot(self._slope)
    
def mean_squared_error(y, predictions):
	return sum(np.square(y-predictions)).mean()
