import numpy as np

def check_dtype(X: np.ndarray) -> np.ndarray:
    if X.dtype != np.float64:
        return X.astype(np.float64)
    
    return X

class MultipleLinearRegression:
    def __init__(self):
        self.intercept = 0
        self.p_dim = 0

    def train(self, X, Y):
        X = check_dtype(X)
        Y = check_dtype(Y)

        Y = np.reshape(Y, (-1, 1))

        self.p_dim = X.shape[1]
        self.__features = np.insert(X, 0, np.ones(X.shape[0]).astype(np.float64), axis=1)

        x_inv = np.linalg.inv(np.dot(self.__features.T, self.__features))

        self.__weights = np.dot(np.dot(x_inv, self.__features.T), Y)
        
        
    def predict(self, newdata):
        assert newdata.shape[1] == self.p_dim, 'Wrong input dimensions'
        newdata = check_dtype(newdata)

        result = []
        for sample in newdata:
            result.append(
                np.dot(sample, self.__weights[1:]) + self.__weights[0]
                )

        return result
