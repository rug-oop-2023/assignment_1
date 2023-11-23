import numpy as np

class MultipleLinearRegression:
    def __init__(self, n_dims:int=1) -> None:
        self.n_dims = n_dims

    def train(self, X:np.ndarray, Y:np.ndarray) -> None:
        assert X.shape[1] == self.n_dims, 'Wrong input dimensions'

        self._features = np.insert(X, 0, np.ones(X.shape[0]), axis=1)

        x_inv = np.linalg.inv(np.dot(self._features.T, self._features))

        self._weights = np.dot(np.dot(x_inv, self._features.T), Y)
        
        
    def predict(self, newdata:np.ndarray) -> np.ndarray:
        assert newdata.shape[1] == self.n_dims, 'Wrong input dimensions'

        result = np.array(len(newdata))
        for sample in newdata:
            result.append(
                np.dot(sample, self._weights[1:]) + self._weights[0]
                )

        return result
