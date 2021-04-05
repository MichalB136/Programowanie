from abc import ABCMeta
from abc import abstractmethod

import numpy as np
from scipy.special import expit, logsumexp

from sklearn.tree._tree import TREE_LEAF
from sklearn.dummy import DummyClassifier

class MyLossFunciton(metaclass=ABCMeta):
    

    is_multi_class = False
    
    def __init__(self, n_classes):
        self.K = n_classes

    def init_estimator(self):
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, y, raw_predictions, sample_weight=None):
        """Compute the loss.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            True labels.

        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves).

        sample_weight : ndarray of shape (n_samples,), default=None
            Sample weights.
        """

    @abstractmethod
    def negative_gradient(self, y, raw_predictions, **kargs):
        """Compute the negative gradient.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            The target labels.

        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.
        """

    def update_terminal_regions(self, tree, X, y, residual, raw_predictions,
                                sample_weight, sample_mask,
                                learning_rate=0.1, k=0):
        """Update the terminal regions (=leaves) of the given tree and
        updates the current predictions of the model. Traverses tree
        and invokes template method `_update_terminal_region`.

        Parameters
        ----------
        tree : tree.Tree
            The tree object.
        X : ndarray of shape (n_samples, n_features)
            The data array.
        y : ndarray of shape (n_samples,)
            The target labels.
        residual : ndarray of shape (n_samples,)
            The residuals (usually the negative gradient).
        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.
        sample_weight : ndarray of shape (n_samples,)
            The weight of each sample.
        sample_mask : ndarray of shape (n_samples,)
            The sample mask to be used.
        learning_rate : float, default=0.1
            Learning rate shrinks the contribution of each tree by
             ``learning_rate``.
        k : int, default=0
            The index of the estimator being updated.

        """   
        #compute leaf for each sample in ``X``.
        terminal_regions = tree.apply(X)

        # mask all which are not in sample mask.
        masked_terminal_regions = terminal_regions.copy()
        masked_terminal_regions[~sample_mask] = -1

        # update each leaf (=perfom line search)
        for leaf in np.where(tree.children_left == TREE_LEAF)[0]:
            self._update_terminal_region(tree, masked_terminal_regions,
                                         leaf, X, y, residual,
                                         raw_predictions[:,k], sample_weight)

        #update predictions (both in-bag and out-of-bag)
        raw_predictions[:, k] += \
            learning_rate * tree.value[:, 0, 0].take(terminal_regions, axis=0)

    @abstractmethod                  
    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, raw_predictions, sample_weight):
        """Template method for updating terminal regions (i.e., leaves)."""

    @abstractmethod
    def get_init_raw_predictions(self, X, estimator):
        """Return the initial raw predictions.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data array.
        estimator : object
            The estimator to use to compute the predictions.

        Returns
        -------
        raw_predictions : ndarray of shape (n_samples, K)
            The initial raw predictions. K is equal to 1 for binary
            classification and regression, and equal to the number of classes
            for multiclass classification. ``raw_predictions`` is casted
            into float64.
        """
        pass

class MyClassificationLossFunction(MyLossFunciton, metaclass=ABCMeta):
    """Base class for classification loss functions. """

    def _raw_prediction_to_proba(self, raw_predictions):
        """Template method to convert raw predictions into probabilities.

        Parameters
        ----------
        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble.

        Returns
        -------
        probas : ndarray of shape (n_samples, K)
            The predicted probabilities.
        """

    @abstractmethod
    def _raw_prediction_to_decision(self, raw_predictions):
        """Template method to convert raw predictions to decisions.

        Parameters
        ----------
        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble.

        Returns
        -------
        encoded_predictions : ndarray of shape (n_samples, K)
            The predicted encoded labels.
        """

    def check_init_estimator(self, estimator):
        """Make sure estimator has fit and predict_proba methods.

        Parameters
        ----------
        estimator : object
            The init estimator to check.
        """
        if not (hasattr(estimator, 'fit') and
                hasattr(estimator, 'predict_proba')):
            raise ValueError(
                "The init parameter must be a valid estimator "
                "and support both fit and predict_proba."
            )

class MyBinomialDeviance(MyClassificationLossFunction):
    """Binomial deviance loss function for binary classification.

    Binary classification is a special case; here, we only need to
    fit one tree instead of ``n_classes`` trees.

    Parameters
    ----------
    n_classes : int
        Number of classes.
    """
    def __init__(self, n_classes):
        if n_classes != 2:
            raise ValueError("{0:s} requires 2 classes; got {1:d} class(es)"
                             .format(self.__class__.__name__, n_classes))
        # we only need to fir one tree for binary clf.
        super().__init__(n_classes=1)

    def init_estimator(self):
        # return the most common class, taking into account the samples
        # weights
        return DummyClassifier(strategy='prior')

    def __call__(self, y, raw_predictions, sample_weight=None):
        """Compute the deviance (= 2 * negative log-likelihood).

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            True labels.

        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble.

        sample_weight : ndarray of shape (n_samples,), default=None
            Sample weights.
        """
        # logaddexp(0, v) == log(1.0 + exp(v))
        raw_predictions = raw_predictions.ravel()
        if sample_weight is None:
            return -2 * np.mean((y * raw_predictions) - 
                                np.logaddexp(0, raw_predictions))
        else:
            return (-2 / sample_weight.sum() * np.sum(
                sample_weight * ((y * raw_predictions) - 
                                 np.logaddexp(0, raw_predictions))))
    
    def negative_gradient(self, y, raw_predictions, **kargs):
        """Compute the residual (= negative gradient).

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            True labels.

        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.
        """
        return y - expit(raw_predictions.ravel())

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, raw_predictions, sample_weight):
        """Make a single Newton-Raphson step.

        our node estimate is given by:

            sum(w * (y - prob)) / sum(w * prob * (1 - prob))

        we take advantage that: y - prob = residual
        """
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

    def get_init_raw_predictions(self, X, estimator):
        probas = estimator.predict_proba(X)
        proba_pos_class = probas[:, 1] 
        eps = np.finfo(np.float32).eps
        proba_pos_class = np.clip(proba_pos_class, eps, 1 - eps)
        # log(x / (1 - x)) is the inverse of the sigmoid (expit) function
        raw_predictions = np.log(proba_pos_class / (1 - proba_pos_class))
        return raw_predictions.reshape(-1, 1).astype(np.float64)  