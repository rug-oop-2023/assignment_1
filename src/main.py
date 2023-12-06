from multiple_linear_regression import MultipleLinearRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


def main():
    model = MultipleLinearRegressor()

    ds = load_diabetes()

    x_train, x_test, y_train, y_test = train_test_split(
        ds.data,
        ds.target,
        test_size=0.2
    )
    model.train(x_train, y_train)

    score = model.evaluate(y_train, y_test)


if __name__ == '__main__':
    main()