import numpy as np


x = np.array([[1, 1], [1, 2], [2, 2], [2, 3],[3,5],[12,16]])
y = np.array([6,3,12,5,6,15])

width = np.size(x,axis=1)
height = np.size(x,axis=0)
newx = np.zeros((height,width+1))
newx[:,0]=1
newx[:,1:] = x
x = newx

firstblock = np.linalg.inv(np.matmul(np.transpose(x),x))

secondblock = np.matmul(np.transpose(x),y)

wstar = np.matmul(firstblock,secondblock)

print(wstar)



from sklearn.linear_model import LinearRegression as linreg

reg = linreg().fit(x,y)

print(reg.coef_)
print(reg.intercept_)