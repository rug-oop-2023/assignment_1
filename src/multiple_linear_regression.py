import numpy as np


class MultipleLinearRegressor:
    def __init__(self, default_weights: np.ndarray = np.empty(1)):
        self._weights = default_weights

    @staticmethod
    def _add_bias_col(matrix: np.ndarray) -> np.ndarray:
        """
        Adds a column vector of ones to a np array to account for the bias value in the slope array.
        :param matrix: a 2d n * p numpy array
        :return: a 2d n * p + n numpy array
        """

        num_samples = matrix.shape[0] if matrix.ndim > 1 else 1
        bias_col = np.ones((num_samples, 1))

        final_matrix = np.hstack((bias_col, matrix)) # error here please fix
        return final_matrix

    @staticmethod
    def _calc_optimal_weights(x_matrix: np.ndarray, y_matrix: np.ndarray) -> np.ndarray:
        """
        Calculates optimal weights using the analytical solution for linear regression.
        :param x_matrix: a 2d n * p + n matrix
        :param y_matrix: a 1d matrix of length n
        :return: a 1d matrix of size p + 1
        """

        # Calculate X^T * X
        x_transpose_x = np.dot(x_matrix.T, x_matrix)

        # Check if X^T * X is invertible
        if np.linalg.det(x_transpose_x) == 0:
            raise ValueError("X^T * X is not invertible. The matrix may not be full rank.")

        # Calculate the inverse of X^T * X
        x_transpose_x_inv = np.linalg.inv(x_transpose_x)

        # Calculate X^T * y
        x_transpose_y = np.dot(x_matrix.T, y_matrix)

        # Calculate the optimal weights
        result = np.dot(x_transpose_x_inv, x_transpose_y)

        return result

    def train(self, observations: np.ndarray, target: np.ndarray) -> None:
        """
        Function to train model - finds analytical solution and adjust the weight values accordingly.

        :param observations: a 2d n * p numpy array
        :param target: a 1d numpy array of length n
        """

        final_obs = self._add_bias_col(observations)
        optimal_weights = self._calc_optimal_weights(final_obs, target)

        self._weights = optimal_weights

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Returns a prediction using formula : ŷ = wX
        :param data:
        :return:
        """
        return np.dot(data, self._weights)

    def evaluate(self, test_data: np.ndarray, true_labels: np.ndarray) -> float:
        """
        Evaluate the model using Mean Squared Error (MSE) as the metric.
        :param test_data: The data to predict.
        :param true_labels: The true labels corresponding to the test data.
        :return: Mean Squared Error (MSE).
        """
        predictions = self.predict(self._add_bias_col(test_data))
        mse = np.mean((true_labels - predictions) ** 2)
        return mse
