import numpy as np

class MultipleLinearRegression:
    def __init__(self):
        self.intercept = 0
        self.p_dim = 0

    def train(self, X, Y):
        if X.dtype != np.float64:
            X = X.astype(np.float64)
        if Y.dtype != np.float64:
            Y = Y.astype(np.float64)

        self.p_dim = X.shape[1]
        self.__weights = np.ones(self.p_dim + 1)
        self.__features = np.insert(X, 0, np.ones(X.shape[0]), axis=1)

        x_inv = np.linalg.inv(np.matmul(np.transpose(self.__features), self.__features))

        self.__weights = np.matmul(x_inv, np.transpose(self.__features))
        self.__weights = np.matmul(self.__weights, Y)

    def get_weights(self):
        return self.__weights
    
    def predict(self, X):
        assert X.shape[1] == self.p_dim, 'Wrong input dimensions'

        result = []
        for data in X:
            temp_result = self.__weights[0]
            
            for i, value in enumerate(data):
                temp_result += value * self.__weights[i+1]
            
            result.append(temp_result)
        
        return result

    def evaluate(self, X, Y):
        predictions = self.predict(X)

        self.mse = (np.square(predictions - Y)).mean(axis=None)

        return self.mse
