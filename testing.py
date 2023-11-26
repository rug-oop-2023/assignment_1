import numpy as np
x = np.array([[2,5,7],[4,190,446]])
y = np.transpose(np.array([4,8]))

dimensions = np.size(x,axis=1)

x_mean = np.zeros(dimensions)
for i in range(dimensions):  
    x_mean[i] = np.mean(x[:,i])

y_mean = np.mean(y,axis=0)

xt = np.transpose(x)
xtone = np.matmul(xt,x)
xtoneinv = np.linalg.inv(xtone)
xttwo = np.matmul(xtoneinv,xt)
wstar = np.matmul(xttwo,y)  ## This is the multidimensional slope

intercept = y_mean- np.matmul(x_mean,wstar)  ## This is the intercept, duh see variable name

print(intercept)
