import numpy as np


class MultipleLinearRegression:
    def __init__(self):
        self._features = 0
        self._intercept = None
        self._slope = None

    def train(self, observations: np.ndarray, target: np.ndarray) -> None:
        '''
        observations is a 2d numpy array of length n rows * p columns
        target is a 1d numpy array of length n
        '''

        self._features = np.shape(observations)[1]

        # obs_mean = observations.mean()
        # target_mean = target.mean()
        # obs_residuals = observations - obs_mean
        # target_residuals = target - target_mean
        obs_residuals = observations
        target_residuals = target

        ones = np.ones((np.shape(observations)[0], 1))
        X = np.hstack((ones, obs_residuals))

        Xt = X.transpose()
        Xi = np.linalg.inv(Xt.dot(X))
        w = Xi.dot(Xt).dot(target_residuals)

        self._intercept = w.item(0)
        self._slope = np.delete(w, 0)

    def predict(self, data: np.ndarray) -> np.ndarray:
        if np.shape(data)[1] != self._features:
            raise Exception("Wrong number of feature columns")
        return self._intercept + data.dot(self._slope)


if __name__ == "__main__":
    from sklearn import datasets

    diabetes = datasets.load_diabetes()
    model = MultipleLinearRegression()
    model.train(diabetes.data, diabetes.target)

    predictions = model.predict(diabetes.data)
    print("Real: ", diabetes.target[:5])
    print("Pred: ", predictions[:5])
