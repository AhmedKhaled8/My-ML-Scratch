import numpy as np

class MultivariateLinearRegression:
    def __init__(self, X, y, include_intercept = False):
        """MultivariateLinearRegression is a class that implements the linear regression algorithm using the gradient
           descent algorithm to generate a linear hypothesis

        Args:
            `X` ([ndarray]): [Features of shape (No. examples, No. Features), preferred to be scaled]
            `y` ([ndarray]): [Target Variable of shape (No. examples, 1), must be continuous variable]
            `include_intercept` (bool, optional): [If True, the class will not include the X0 feature\\
                representing the intercept (which means you included it). Else, it will]. Defaults to False.
        """
        assert X.shape[0] == y.shape[0], "X and y must have same number of rows (samples)"
        assert len(y.shape) == 2, "y doesn't have a valid shape. Use y.reshape(-1, 1)"
        self.m = self.X.shape[0]
        if not include_intercept:
            self.X = np.hstack((np.ones((self.m, 1)), X))
        else:
            self.X = X
        self.y = y
        self.n = self.X.shape[1]
    
    def __calculate_hypothesis(self, X):
        """calculates the output of the hypothesis (predicted values) using current parameters

        Args:
            `X` ([ndarray]): [Features Matrix]

        Returns:
            [ndarray]: [the output of the hypothesis in shape (No. examples, 1)]
        """
        h = X @ self.theta
        return h
    
    def __calculate_cost(self, h):
        """calculates the cost between the output of the hypothesis and the true values `y`

        Args:
            h ([ndarray]): [the output of the hypothesis]

        Returns:
            [float]: [a single number representing the cost]
        """
        cost = (1 / (2 * self.m)) * (h - self.y).T @ (h - self.y)
        return cost[0][0] # cost is an array of shape[0][0]

    def __calculate_grads(self, h):
        """calculates the gradients by which the parameters will be updated

        Args:
            h ([ndarray]): [the output of the hypothesis]

        Returns:
            [ndarray]: [the gradients for the parameters in shape (No. parameters `n`, 1)]
        """
        grads = (1 / self.m) * self.X.T @ (h - self.y)
        return grads

    def __update_theta(self, grads):
        """updates the parameters of the hypothesis by subtracting the current parameters from the gradients
        weighted by `grads`

        Args:
            grads (ndarray): the gradients for the parameters in shape (No. parameters `n`, 1)
        """
        self.theta = self.theta - self.alpha * grads

    def fit(self, alpha = 0.01, n_iterations = 100, log = False):
        self.theta = np.zeros((self.n, 1))
        self.alpha = alpha
        self.cost = [0] * n_iterations
        for it in range(n_iterations):
            # calculate the predicted values
            h = self.__calculate_hypothesis(self.X)
            # calculate the cost
            cost = self.__calculate_cost(h)
            # log the cost for each iteration
            if log:
                print(f"Iteration {it + 1}/{n_iterations}\t\t\tCost: {cost}")
            # assign the cost to the cost history
            self.cost[it] = cost
            # calculate the gradients
            grads = self.__calculate_grads(h)
            # update the parameters
            self.__update_theta(grads)

    def predict(self, X):
        """calculates the predicted values using the fitted parameters

        Args:
            X (ndarray): the features matrix

        Returns:
            [ndarray]: the predicted values of shape (X.shape[0], 1)
        """
        X_intercept = np.hstack((np.ones((X.shape[0], 1)), X))
        return self.__calculate_hypothesis(X_intercept)

    def __calculate_rmse(self, true, predicted):
        """calculates the rmse error between the true and predicted values

        Args:
            true (ndarray): the true values
            predicted (ndarray): the predicted valyes

        Returns:
            ndarray: array containting a single number representing the error
        """
        error = np.sqrt(np.sum(np.power(true - predicted, 2), axis = 0) / len(true))
        return error

    def __calculate_r2(self, true, predicted):
        """calculates the R2 metric between the true and predicted values

        Args:
            true (ndarray): the true values
            predicted (ndarray): the predicted valyes

        Returns:
            ndarray: array containting a single number representing the R2 value
        """
        ss_res = np.sum(np.power(true - predicted, 2), axis = 0)
        ss_tot = np.sum(np.power(true - np.mean(true), 2), axis = 0)
        return 1 - (ss_res / ss_tot)

    def score(self, true, predicted, criterion = "r2"):
        """calculates the score between the true and predicted values using given criterion

        Args:
            true (ndarray): the true values
            predicted (ndarray): the predicted values
            criterion (str, optional): the criterion of the score. Defaults to "r2".

        Returns:
            ndarray: array containting a single number representing the score value
        """
        if criterion == "r2":
            return self.__calculate_r2(true, predicted)
        elif criterion == "rmse":
            return self.__calculate_rmse(true, predicted)

