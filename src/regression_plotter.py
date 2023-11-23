import matplotlib.pyplot as plt

class ReggresionPlotter:
    def __init__(self, model):
        self.model = model
    
    def plot_singular_feature(self, X, Y):
        assert X.shape[1] == 1 or X.shape[1] == 0, 'Wrong input dimensions'

        X = X.reshape((-1, 1))
        Y = Y.reshape((-1, 1))

        predictions = self.model.predict(X)

        plt.plot(X, predictions ,color='k')
        plt.scatter(X, Y, color='r')
        plt.show()
    
    def plot_two_features(self, X, Y):
        assert X.shape[1] == 2, 'Wrong input dimensions'

        Y = Y.reshape((-1, 1))

        predictions = self.model.predict(X)

        plt.plot(X, predictions ,color='k', projection='3d')
        plt.show()