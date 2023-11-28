import numpy as np

class SimpleLinearRegressor:
    def __init__(self, dimension, default_intercept, default_slope):
        self.slope = np.zeros(dimension)
        self.intercept = default_intercept

    def train(self, x, y):
        '''
        x is a matrix with n rows(number of data points) and p columns(number of dimensions)
        y is a 1D array that contains the predictions
        '''
        dimensions = np.size(x,axis=1)
        
        ''''
        # x_mean is an array of length p, containing the mean values of the data points for each feature
        x_mean = np.zeros(dimensions)
        for i in range(dimensions):  
            x_mean[i] = np.mean(x[:,i])

        y_mean = np.mean(y,axis=0)     
        '''
        xt = np.transpose(x) 
        xtone = np.matmul(xt,x)
        xtoneinv = np.linalg.inv(xtone)
        xttwo = np.matmul(xtoneinv,xt)
        
        self.slope = np.matmul(xttwo,y) 
        self.intercept = self.slope[0]
    
    def predict(self, x):
        return np.matmul(x,self.slope)
    
if __name__ == "__main__":
  
    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import LinearRegression as linreg
    from sklearn.metrics import mean_squared_error

    diabetes = load_diabetes()
    x=diabetes.data
    y=diabetes.target

    '''
    x = np.array([[1, 1], [1, 2], [2, 2], [2, 3],[3,5],[12,16]])
    y = np.array([6,3,12,5,6,15])
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    #adding noise
    x += np.random.rand(*x.shape)
    '''
    
    #adding column of ones to the x matrix
    width = np.size(x,axis=1)
    height = np.size(x,axis=0)
    newx = np.zeros((height,width+1))
    newx[:,0]=1
    newx[:,1:] = x
    x = newx

    model = SimpleLinearRegressor()
    reg = linreg().fit(x,y)

    model.train(x, y)
    model_pred = model.predict(x)
    scikit_pred = reg.predict(x)

    mse_model = mean_squared_error(y, model_pred)
    mse_scikit = mean_squared_error(y, scikit_pred)
    print(x[:5])
    print(f"SimpleLinerRegressor coefficients --intercept {model.intercept} --slope {model.slope}")
    print("MSE model:", mse_model)
    print(f"scikit coefficients --intercept {reg.intercept_} --slope {reg.coef_}")
    print("MSE scikit:", mse_scikit)