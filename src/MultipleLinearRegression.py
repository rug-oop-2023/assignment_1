import pandas as pd
import numpy as np

"""class SimpleLinearRegressor:
    def __init__(self):
        self.intercept = 0
        self.slope = 0
model = SimpleLinearRegressor()"""

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