import numpy as np

class SimpleLinearRegressor:
    def __init__(self, default_intercept=0, default_slope=0):
        self.intercept = default_intercept
        self.slope = default_slope
    
    def train(self, x, y):
        '''
        x and y are two 1D numpy arrays of the same length
        '''
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        slope_numerator = np.sum((x - x_mean) * (y - y_mean))
        slope_denominator = np.sum((x - x_mean)**2)
        self.slope = slope_numerator / slope_denominator
        self.intercept = y_mean - self.slope * x_mean
    
    def predict(self, x):
        '''
        x is a numpy array
        '''
        return self.slope * x + self.intercept
    
if __name__ == "__main__":
    model = SimpleLinearRegressor(0,0)
    x = np.array([1,2,3,4,5,6])
    y = np.array([0,1,2,3,4,5])
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x += np.random.rand(*x.shape)
    model.train(x, y)
    #x += np.random.rand(*x.shape)
    y_pred = model.predict(x)
    print(f"SimpleLinerRegressor coefficients -- intercept {model.intercept} -- slope {model.slope}")
    print("Ground truth and predicted values:", y, y_pred, sep="\n")