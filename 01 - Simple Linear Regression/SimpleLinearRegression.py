import numpy as np

class SimpleLinearRegression:
    """
    ### Description:
    ---
    `SimpleLinearRegression` is a class that can be used to find the relationship between a single contious feature and a contious target.

    ### Class Members:
    ---
    `b0`: the intercept of the y-axis at x = 0
    `b1`: the slope of the line (the derivative with respect to the feature)

    ### APIs:
    ---
    `fit(X, y)`: calculate `b0` and `b1` for a single feature input `X` and a target `y`
    `predict(X)`: calculate predicted output for given input `X` using `b0` and `b1` previously calculated from `fit`
    `score(true, predicted, criterion)`: calculate the root mean square error between two values usually `true` and `predicted` using the metric of `criterion`

    """
    def __init__(self):
        """
        """
        self.b0 = 0.0
        self.b1 = 0.0

    def __mean(self, x):
        return np.mean(x, axis=0)
    
    def __variance(self, x):
        x_mean = self.__mean(x)
        x_var = np.sum(np.power(x - x_mean, 2), axis = 0)
        return x_var
    
    def __covariance(self, x, y):
        mean_x = self.__mean(x)
        mean_y = self.__mean(y)
        covariance = np.sum(np.multiply(x - mean_x, y - mean_y))
        return covariance

    def __calculate_rmse(self, true, predicted):
        error = np.sqrt(np.sum(np.power(true - predicted, 2), axis = 0) / len(true))
        return error

    def __calculate_r2(self, true, predicted):
        ss_res = np.sum(np.power(true - predicted, 2), axis = 0)
        ss_tot = np.sum(np.power(true - self.__mean(true), 2), axis = 0)
        return 1 - (ss_res / ss_tot)
    
    def fit(self, X, y):
        assert X.shape[1] == 1, "X must include only one feature. Try X.reshape(-1, 1)"
        assert y.shape[1] == 1, "y must include only one feature. Try y.reshape(-1, 1)"
        self.b1 = self.__covariance(X, y) / self.__variance(X)
        self.b0 = self.__mean(y) - self.b1 * self.__mean(X)
        return self.b0, self.b1
    
    def predict(self, X):
        assert X.shape[1] == 1, "X must include only one feature. Try X.reshape(-1, 1)"
        return self.b0 + self.b1 * X

    def score(self, true, predicted, criterion = "r2"):
        if criterion == "r2":
            return self.__calculate_r2(true, predicted)
        elif criterion == "rmse":
            return self.__calculate_rmse(true, predicted)
        