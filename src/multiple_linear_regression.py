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

        self.__weights = np.dot(x_inv, np.dot(self.__features.T, Y))

    def get_weights(self):
        return self.__weights

    def get_features(self):
        return self.__features
    
    def predict(self, X):
        assert X.shape[1] == self.p_dim, 'Wrong input dimensions'
        X = check_dtype(X)

        result = []
        for data in X:
            temp_result = self.__weights[0]
            
            for i, value in enumerate(data):
                temp_result += value * self.__weights[i+1]
            
            result.append(temp_result)
        
        return result

    def evaluate(self, X, Y):
        predictions = self.predict(X)

        self.mse = (np.square(predictions - Y)).mean()

        return self.mse
