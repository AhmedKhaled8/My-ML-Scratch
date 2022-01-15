import numpy as np

class BinaryLogisticRegression:
    def __init__(self, X, y, include_intercept = False):
        """BinaryLogisticRegression is a class that implements the logistic regression algorithm used for binary classification problems
           to have a decision boundary between the two classes

        Args:
            `X` ([ndarray]): [Features of shape (No. examples, No. Features), preferred to be scaled]
            `y` ([ndarray]): [Target Variable of shape (No. examples, 1), must be continuous variable]
            `include_intercept` (bool, optional): [If True, the class will not include the X0 feature\\
                representing the intercept (which means you included it). Else, it will]. Defaults to False.
        """
        assert X.shape[0] == y.shape[0], "X and y must have same number of rows (samples)"
        assert len(y.shape) == 2, "y doesn't have a valid shape. Use y.reshape(-1, 1)"
        self.m = X.shape[0]
        if not include_intercept:
            self.X = np.hstack((np.ones((self.m, 1)), X))
        else:
            self.X = X
        self.y = y
        
        self.n = self.X.shape[1]

    def __sigmoid(self, X):
        """calculate the sigmoid to a value / an array"""
        return 1 / (1 + np.exp(-X))
    
    def __calculate_hypothesis(self, X):
        """calculates the output of the hypothesis (predicted values) using current parameters

        Args:
            `X` ([ndarray]): [Features Matrix]

        Returns:
            [ndarray]: [the output of the hypothesis in shape (No. examples, 1)]
        """
        h = self.__sigmoid(X @ self.theta)
        return h
    
    def __calculate_likelihood(self, h):
        """calculates the likelihood of the parameters between the output of the hypothesis and the true values `y`

        Args:
            `h` ([ndarray]): [the output of the hypothesis]

        Returns:
            [float]: [a single number representing the cost]
        """
        likelihood = 1 / self.m * ((self.y.T @ np.log(h)) + ((np.ones(self.y.shape) - self.y).T @ (np.log(np.ones(h.shape) - h))))
        return likelihood[0][0] # cost is an array of shape[0][0]

    def __calculate_grads(self, h):
        """calculates the gradients by which the parameters will be updated

        Args:
            `h` ([ndarray]): [the output of the hypothesis]

        Returns:
            [ndarray]: [the gradients for the parameters in shape (No. parameters `n`, 1)]
        """
        grads = (1 / self.m) * self.X.T @ (self.y - h)
        return grads

    def __update_theta(self, grads):
        """updates the parameters of the hypothesis by adding the current parameters from the gradients
        weighted by `grads` for maximizing the likelihood

        Args:
            `grads` (ndarray): the gradients for the parameters in shape (No. parameters `n`, 1)
        """
        self.theta = self.theta + self.alpha * grads

    def fit(self, alpha = 0.01, n_iterations = 100, log = False):
        """trains the model based on the data given at initialization

        Args:
            `alpha` (float, optional): the learning rate which is multiplied with the gradients. Defaults to 0.01.
            `n_iterations` (int, optional): the number of iterations (epochs) used for training. Defaults to 100.
            `log` (bool, optional): if true, a log message with the likelihood for each iteration will be printed. Defaults to False.
        """
        self.theta = np.zeros((self.n, 1))
        self.alpha = alpha
        self.cost = [0] * n_iterations
        for it in range(n_iterations):
            # calculate the predicted values
            h = self.__calculate_hypothesis(self.X)
            # calculate the cost
            likelihood = self.__calculate_likelihood(h)
            # log the likelihood for each iteration
            if log:
                print(f"Iteration {it + 1}/{n_iterations}\t\t\tCost: {likelihood}")
            # assign the cost to the cost history
            self.cost[it] = likelihood
            # calculate the gradients
            grads = self.__calculate_grads(h)
            # update the parameters
            self.__update_theta(grads)

    def predict(self, X):
        """calculates the predicted values using the fitted parameters

        Args:
            `X` (ndarray): the features matrix

        Returns:
            [ndarray]: the predicted values of shape (X.shape[0], 1)
        """
        X_intercept = np.hstack((np.ones((X.shape[0], 1)), X))
        return self.__calculate_hypothesis(X_intercept)

    def score(self, y_true, y_predicted, probabilistic=False, criterion="acc"):
        """returns the score between the true and predicted values based on given criterion.

        Args:
            `y_true` (ndarray): the true class
            `y_predicted` (ndarray): the predicted class / class-probabilities
            `probabilistic` (bool, optional): if true, the `y_predicted` is mapped to either 1 / 0. if false, it will be mapped inside this function. Defaults to False.
            `criterion` (str, optional): the criterion of the score. Defaults to "acc".
        """
        y_predicted_used = None
        if probabilistic:
            y_predicted_used = y_predicted.copy()
            y_predicted_used = (y_predicted_used >= 0.5).astype(np.int64)
        else:
            y_predicted_used = y_predicted.copy()
        
        true_positive = np.sum((y_true + y_predicted_used == 2).astype(np.int64))
        true_negative = np.sum((y_true + y_predicted_used == 0).astype(np.int64))
        false_positive = np.sum((np.logical_and(y_true == 0, y_predicted_used == 1)).astype(np.int64))
        false_negative = np.sum((np.logical_and(y_true == 1, y_predicted_used == 0)).astype(np.int64))
        if criterion == "acc":
            return (true_positive + true_negative)/(true_positive + true_negative + false_negative + false_positive)
