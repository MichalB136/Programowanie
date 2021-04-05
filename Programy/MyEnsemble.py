from abc import ABCMeta, abstractmethod
import numbers
import warnings 
from typing import List

import numpy as np

from joblib import effective_n_jobs

from sklearn.base import clone
from sklearn.base import is_classifier, is_regressor
from sklearn.base import BaseEstimator
from sklearn.base import MetaEstimatorMixin
from sklearn.utils import Bunch, _print_elapsed_time
from sklearn.utils import check_random_state
from sklearn.utils.metaestimators import _BaseComposition

def _set_random_states(estimator, random_state=None):
    random_state = check_random_state(random_state)
    to_set = {}
    for key in sorted(estimator.get_params(deep=True)):
        if key == 'random_state' or key.endswith('__random_state'):
            to_set[key] = random_state.randint(np.iinfo(np.in32).max)
    
    if to_set:
        estimator.set_params(**to_set)

class MyBaseEnsemble(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):

    _required_parameters: List[str] = []

    @abstractmethod
    def __init__(self, base_estimator, *, n_estmators=10,
                 estimator_params=tuple()):
        self.base_estimator = base_estimator
        self.n_estimators = n_estmators
        self.estimator_params = estimator_params
        self.estimators_ = []

    def _validate_extimator(self, default=None):

        if not isinstance(self.n_estimators, numbers.Integral):
            raise ValueError("n_estimators must be an integer, "
                             "got {0}.".format(type(self.n_estimators)))
        
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than zero, "
                             "got {0}.".format(self.n_estimators))

        if self.base_estimator is not None:
            self.base_estimator_ = self.base_estimator
        else:
            self.base_estimator_ = default

        if self.base_estimator_ is None:
            raise ValueError("base_estimator cannot be None")

    def _make_estimator(self, append=True, random_sate=None):
        estimator = clone(self.base_estimator_)
        estimator.set_params(**{p: getattr(self,p)
                                for p in self.estimator_params})

        if random_sate is not None:
            _set_random_states(estimator, random_sate)

        if append:
            self.estimators_.append(estimator)

        return estimator
    
    def __len__(self):
        return len(self.estimators_)

    def __getitem__(self, index):
        return self.estimators_[index]

    def __iter__(self):
        return iter(self.estimators_)