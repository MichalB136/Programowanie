from abc import ABCMeta
from abc import abstractmethod

import numpy as np
from scipy.special import expit
from sklearn.tree._tree import TREE_LEAF
from sklearn.dummy import DummyClassifier, DummyRegressor


class LeastSquareLossFunction():

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


    def get_init_raw_prediciton(self, X, estimator):
        predictions = estimator.predict(X)
        return predictions.reshape(-1,1).astype(np.float64)

    def update_terminal_regions(self, tree, X, y, residual, raw_predicitons,
                                sample_weight, sample_mask,
                                learning_rate=0.1, k=0):
        raw_predicitons[:, k] += learning_rate * tree.predict(X).ravel()

class BinomialDeviance():
    
    is_multi_class = False
    
    def __init__(self, n_classes):
        if n_classes != 2:
            raise ValueError("{0:s} requires 2 classes; got {1:d} class(es)"
                             .format(self.__class__.__name__, n_classes))       
        n_classes = 1
        self.K = n_classes

    def init_estimator(self):

        return DummyClassifier(strategy='prior')

    def __call__(self, y, raw_predictions, sample_weight=None):

        raw_predictions = raw_predictions.ravel()
        if sample_weight is None:
            return -2 * np.mean((y * raw_predictions) -
                                np.logaddexp(0, raw_predictions))
        else:
            return (-2 / sample_weight.sum() * np.sum(
                sample_weight * ((y * raw_predictions) - 
                                 np.logaddexp(0, raw_predictions))))

    def negative_gradient(self, y, raw_predictions, **kargs):

        return y - expit(raw_predictions.ravel())

    def update_terminal_regions(self, tree, X, y, residual, raw_predictions,
                                sample_weight, sample_mask,
                                learning_rate=0.1, k=0):
        terminal_regions = tree.apply(X)

        masked_termianl_regions = terminal_regions.copy()
        masked_termianl_regions[~sample_mask] = -1
        for leaf in np.where(tree.children_left == TREE_LEAF)[0]:
            self._update_terminal_region(tree, masked_termianl_regions,
                                         leaf, X, y, residual, 
                                         raw_predictions[:, k], sample_weight)
        
        raw_predictions[:, k] += \
            learning_rate * tree.value[:, 0, 0].take(terminal_regions, axis=0)

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, raw_predictions, sample_weight):

        terminal_region = np.where(terminal_regions == leaf)[0]
        residual = residual.take(terminal_region, axis=0)
        y = y.take(terminal_region, axis=0)
        sample_weight = sample_weight.take(terminal_region, axis=0)

        numerator = np.sum(sample_weight * residual)
        denominator = np.sum(sample_weight *
                             (y - residual) * (1 - y + residual))

        # prevents overflow and division by zero
        if abs(denominator) < 1e-150:
            tree.value[leaf, 0, 0] = 0.0
        else:
            tree.value[leaf, 0, 0] = numerator / denominator

    def _raw_prediction_to_proba(self, raw_predictions):

        proba = np.ones((raw_predictions.shape[0], 2), dtype=np.float64)
        proba[:, 1] = expit(raw_predictions.ravel())
        proba[:, 0] -= proba[:, 1]
        return proba

    def _raw_prediction_to_decision(self, raw_predictions):

        proba = self._raw_prediction_to_proba(raw_predictions)
        return np.argmax(proba, axis=1)

    def get_init_raw_prediciton(self, X, estimator):
        
        probas = estimator.predict_proba(X)
        proba_pos_class = probas[:, 1]
        eps = np.finfo(np.float32).eps
        proba_pos_class = np.clip(proba_pos_class, eps, 1 - eps)
        # log(x / (1 - x)) is the inverse of the sigmoid (expit) functin
        raw_predictions = np.log(proba_pos_class / (1 - proba_pos_class))
        return raw_predictions.reshape(-1, 1).astype(np.float64)