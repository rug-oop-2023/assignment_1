from multiple_linear_regression import MultipleLinearRegression
import matplotlib.pyplot as plt

class ReggresionPlotter:
    def __init__(self):
        self.status = 1
    
    def plot_singular_feature(self, X, Y):
        assert X.shape[1] == 1 or X.shape[1] == 0, 'Wrong input dimensions'

        X = X.reshape((-1, 1))
        Y = Y.reshape((-1, 1))

        model = MultipleLinearRegression()
        model.train(X, Y)
        predictions = model.predict(X)

        plt.plot(X, predictions ,color='k')
        plt.scatter(X, Y, color='r')
        plt.show()
        