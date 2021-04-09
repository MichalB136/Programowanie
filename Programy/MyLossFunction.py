from abc import ABCMeta
from abc import abstractmethod

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor


class MeanSquareLossFunction():

    def __init__(self, n_classes):
        if n_classes != 1:
            raise ValueError("``n_classes`` must be 1 for regression but "
                             "was %r" % n_classes)
        self.K = n_classes

    def init_estimator(self):
        return DummyRegressor(strategy='mean')

    def __call__(self, y, raw_predictions, sample_weight=None):
        raw_predictions = raw_predictions.ravel()
        if sample_weight is None:
            return np.mean((y - raw_predictions.ravel()) ** 2)
        else:
            return (1 / sample_weight.sum() * np.sum(
                sample_weight * ((y - raw_predictions.ravel()) ** 2)))

    def negative_gradient(self, y, raw_predicitons, **kargs):
        return y - raw_predicitons.ravel()

    def compute_gamma(self, tree, X, y, gamma, raw_predicitons_copy):
        test = raw_predicitons_copy.ravel() + gamma.ravel() * tree.predict(X).ravel()
        test2 = -y + test
        # print(sum(test2))
        return sum(test2)

    def get_init_raw_prediciton(self, X, estimator):
        predictions = estimator.predict(X)
        return predictions.reshape(-1,1).astype(np.float64)

    def update_terminal_regions(self, tree, X, y, gamma, residual, raw_predicitons,
                                sample_weight, sample_mask,
                                learning_rate=0.1, k=0):
        raw_predicitons[:, k] += learning_rate * tree.predict(X).ravel()