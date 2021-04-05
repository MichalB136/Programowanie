import numpy as np
from sklearn.tree import DecisionTreeRegressor

class SquareLossFunction():

    def __init__(self, n_classes):
        self.K = n_classes

    def __call__(self, y, raw_predictions, sample_weight=None):
        raw_predictions = raw_predictions.ravel()
        if sample_weight is None:
            return -2 * sum((y - raw_predictions))

    def get_init_raw_prediciton(self, y):
        mean_y = np.mean(y)
        constant_val = [-2 * sum(i - mean_y) for i in y]
        constant_val = sum(constant_val)
        return constant_val



class MyGradientBoosting():

    def __init__(self, loss='square', learning_rate=0.1, max_depth=3, 
                 n_estimators=100, n_classes=1):
        self.learnig_rate = learning_rate
        self.loss = loss
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.n_classes = n_classes

    def get_loss_func(self):
        if self.loss == 'square':
            loss_class = SquareLossFunction
            self.loss_ = loss_class(self.n_classes)

    def initialize(self, X, y):
        # inicjalizacja procesu
        self.get_loss_func()
        self.estimators = np.zeros(X.shape[0])
        self.estimators[0] = self.loss_.get_init_raw_prediciton(y.values)

    def pseudo_res(self, i, y):
        p_val = self.estimators[i - 1]
        #Compute pseudo-residual
        residual = [-(self.loss_(y_val, p_val)) for y_val in y.values] 
        return residual

    def fit_stage(self, X, y, learning_rate, max_depth):
        #zrob drzewka jako pseuda residuale
        pass

        