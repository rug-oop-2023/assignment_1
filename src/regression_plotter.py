import matplotlib.pyplot as plt
import numpy as np

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
    
    def plot_two_features(self, X, Y, predictions):
        assert X.shape[1] == 2, 'Wrong input dimensions'

        Y = Y.reshape((-1, 1))

        #predictions = self.model.predict(X)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(X[:, 0], X[:, 1], Y, c='r', marker='o', label='Actual data')

        ax.scatter(X[:, 0], X[:, 1], predictions, c='b', marker='^', label='Predicted data')

        x1_range = np.arange(X[:, 0].min(), X[:, 0].max())
        x2_range = np.arange(X[:, 1].min(), X[:, 1].max())
        x1_plane, x2_plane = np.meshgrid(x1_range, x2_range)
        
        predictions_reshaped = predictions.reshape(x1_plane.shape)

        ax.plot_surface(x1_plane, x2_plane, predictions_reshaped, alpha=0.5)

        plt.show()

    def plot_multiple_features(self, X, Y):
        predictions = self.model.predict(X)
        for col in range(self.model.n_dims - 1):
            temp_arr = X[:, [col, col+1]]

            self.plot_two_features(temp_arr, Y, np.array(predictions))