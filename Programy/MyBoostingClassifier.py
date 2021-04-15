import numbers
import numpy as np

from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import issparse

from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils import column_or_1d, check_array, check_random_state
from sklearn.ensemble._base import BaseEnsemble

from time import time

from MyLossFunction import LeastSquareLossFunction


class VerboseReporter:
    """Reports verbose output to stdout.

    Parameters
    ----------
    verbose : int
        Verbosity level. If ``verbose==1`` output is printed once in a while
        (when iteration mod verbose_mod is zero).; if larger than 1 then output
        is printed for each update.
    """
    def __init__(self, verbose):
        self.verbose = verbose

    def init(self, est, begin_at_stage=0):
        """Initialize reporter

        Parameters
        ----------
        est : Estimator
            The estimator

        begin_at_stage : int, default=0
            stage at which to begin reporting
        """
        # header fields and line format str
        header_fields = ['Iter', 'Train Loss']
        verbose_fmt = ['{iter:>10d}', '{train_score:>16.4f}']
        # do oob?
        if est.subsample < 1:
            header_fields.append('OOB Improve')
            verbose_fmt.append('{oob_impr:>16.4f}')
        header_fields.append('Remaining Time')
        verbose_fmt.append('{remaining_time:>16s}')

        # print the header line
        print(('%10s ' + '%16s ' *
               (len(header_fields) - 1)) % tuple(header_fields))

        self.verbose_fmt = ' '.join(verbose_fmt)
        # plot verbose info each time i % verbose_mod == 0
        self.verbose_mod = 1
        self.start_time = time()
        self.begin_at_stage = begin_at_stage

    def update(self, j, est):
        """Update reporter with new iteration.

        Parameters
        ----------
        j : int
            The new iteration
        est : Estimator
            The estimator
        """
        do_oob = est.subsample < 1
        # we need to take into account if we fit additional estimators.
        i = j - self.begin_at_stage  # iteration relative to the start iter
        if (i + 1) % self.verbose_mod == 0:
            oob_impr = est.oob_improvement_[j] if do_oob else 0
            remaining_time = ((est.n_estimators - (j + 1)) *
                              (time() - self.start_time) / float(i + 1))
            if remaining_time > 60:
                remaining_time = '{0:.2f}m'.format(remaining_time / 60.0)
            else:
                remaining_time = '{0:.2f}s'.format(remaining_time)
            print(self.verbose_fmt.format(iter=j + 1,
                                          train_score=est.train_score_[j],
                                          oob_impr=oob_impr,
                                          remaining_time=remaining_time))
            if self.verbose == 1 and ((i + 1) // (self.verbose_mod * 10) > 0):
                # adjust verbose frequency (powers of 10)
                self.verbose_mod *= 10


class MyGradientBoosting(BaseEnsemble):

    def __init__(self, *,loss='square', learning_rate=0.1, max_depth=3, 
                 n_estimators=100, criterion='friedman_mse', min_samples_split=2,
                 init=None,
                 min_samples_leaf=1, min_weight_fraction_leaf=0,
                 min_impurity_decrease=0, min_impurity_split=None, max_features=None, 
                 random_state=None, ccp_alpha=0.0, verbose=0, max_lead_nodes=None):
        self.learning_rate = learning_rate
        self.loss = loss
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.max_features = max_features
        self.max_leaf_nodes = max_lead_nodes
        self.random_state = random_state
        self.ccp_alpha = ccp_alpha
        self.init = init        
        # self.subsample = subsample
        self.verbose = verbose
        # self.n_iter_no_change = n_iter_no_change

    def _initialize(self, X, y):
        # inicjalizacja procesu
        if self.loss == 'square':
            loss_class = LeastSquareLossFunction
            self.loss_ = loss_class(self.n_classes_)
        self.init_ = self.init
        if self.init_ is None:
            self.init_ = self.loss_.init_estimator()

        self.train_score_ = np.zeros((self.n_estimators,), dtype=np.float64)    
        self.estimators_ = np.empty((self.n_estimators, self.loss_.K), dtype=np.object)
        self.gamma_ = np.zeros(self.n_estimators) 
        self.gamma = np.zeros((self.n_estimators, y.shape[0]), dtype=np.float64)
        self.estimators = np.zeros((self.n_estimators, X.shape[0]))
    
    def _fit_stage(self, i, X, y, raw_predictions, sample_weight, sample_mask,
                   random_state, X_idx_sorted, X_csc=None, X_csr=None):
        assert sample_mask.dtype == np.bool
        loss = self.loss_
        # original_y = y

        raw_predictions_copy = raw_predictions.copy()
        
        for k in range(loss.K):
            #2.1
            residual = loss.negative_gradient(y, raw_predictions_copy, k=k,
                                              sample_weight=sample_weight)
            #2.2
            tree = DecisionTreeRegressor(
                criterion=self.criterion,
                splitter='best',
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                min_impurity_split=self.min_impurity_split,
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                random_state=random_state,
                ccp_alpha=self.ccp_alpha)
            X = X_csr if X_csr is not None else X
            tree.fit(X, residual, sample_weight=sample_weight,
                     check_input=False, X_idx_sorted=X_idx_sorted)
            #2.4

            loss.update_terminal_regions(
                tree.tree_, X, y, residual, raw_predictions, sample_weight,
                sample_mask, learning_rate=self.learning_rate, k=k)      
            # print(raw_predictions)
            # print("--")
            self.estimators_[i, k] = tree    
       
        return raw_predictions

    def _check_params(self):
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0 but "
                             f"was {self.n_estimators}")

        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be greather than 0 but "
                             f"was {self.learning_rate}")

        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                # if is_classification
                if self.n_classes_ > 1:
                    max_features = max(1, int(np.sqrt(self.n_features_)))
                else:
                    # is regression
                    max_features = self.n_features_
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError("Invalid value for max_features: %r. "
                                 "Allowed string values are 'auto', 'sqrt' "
                                 "or 'log2'." % self.max_features)
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        else: # float
            if 0. < self.max_features <= 1.:
                max_features = max(int(self.max_features * 
                                       self.n_features_), 1)
            else:
                raise ValueError("max_features must be in (0, n_features]")

        self.max_features_ = max_features


    def _clear_state(self):
        if hasattr(self, 'estimators_'):
            self.estimators_ = np.empty((0,0), dtype=np.object)
        if hasattr(self, 'train_score_'):
            del self.train_score_
        if hasattr(self, 'init_'):
            del self.init_
        if hasattr(self, '_rng'):
            del self._rng

    def _is_initialized(self):
        return len(getattr(self, 'estimators_', [])) > 0 

    def fit(self, X, y, sample_weight=None):

        self._clear_state()

        X, y = self._validate_data(X, y, accept_sparse=['csr', 'csc', 'coo'],
                                   dtype=np.float32, multi_output=True)
        n_samples, self.n_features_ = X.shape

        sample_weight_is_none = sample_weight is None

        sample_weight = _check_sample_weight(sample_weight, X)

        y = column_or_1d(y, warn=True)
        y = self._validate_y(y, sample_weight)

        self._check_params()
        
        if not self._is_initialized():

            self._initialize(X, y)

            if self.init_ == 'zero':
                raw_predictions = np.zeros(shape=(X.shape[0], self.loss_.K),
                                        dtype=np.float64)
            else:
                if sample_weight_is_none == None:
                    self.init_.fit(X,y)
                else:
                    msg = ("The initial estimator {} does not support sample "
                           "weights.".format(self.init_.__class__.__name__))
                    try:
                        self.init_.fit(X, y, sample_weight=sample_weight)
                    except TypeError:
                        raise ValueError(msg)
                    except ValueError as e:
                        if "pass parameters to specific steps of "\
                           "your pipeline using the "\
                           "stepname__parameter" in str(e):  # pipeline
                            raise ValueError(msg) from e
                        else:  # regular estimator whose input checking failed
                            raise
                # 1.
                raw_predictions = self.loss_.get_init_raw_prediciton(X, self.init_)
                self.gamma = raw_predictions.copy()

            begin_at_stage = 0
            self._rng = check_random_state(self.random_state)
        # print(raw_predictions)
        X_idx_sorted = None

        n_stages = self._fit_stages(
            X, y, raw_predictions, sample_weight, self._rng, 
            begin_at_stage, X_idx_sorted)

        self.n_estimators_ = n_stages
        return self

    def _fit_stages(self, X, y, raw_predictions, sample_weight, random_state,
                    begin_at_stage=0, X_idx_sorted=None):

        n_samples = X.shape[0]
        # do_oob = self.subsample < 1.0
        sample_mask = np.ones((n_samples, ), dtype=np.bool)
        # n_inbag = max(1, int(self.subsample * n_samples))
        loss_ = self.loss_

        if self.verbose:
            verbose_reporter = VerboseReporter(verbose=self.verbose)
            verbose_reporter.init(self, begin_at_stage)

        X_csc = csc_matrix(X) if issparse(X) else None
        X_csr = csr_matrix(X) if issparse(X) else None

        i = begin_at_stage
        for i in range(begin_at_stage, self.n_estimators):

            raw_predictions = self._fit_stage(
                i, X, y, raw_predictions, sample_weight, sample_mask,
                random_state, X_idx_sorted, X_csc, X_csr)
            
            self.train_score_[i] = loss_(y, raw_predictions, sample_weight)
        return i + 1

    def _validate_y(self, y, sample_weight):

        self.n_classes_ = 1
        if y.dtype.kind == 'O':
            y = y.astype(np.flout64)
        
        return y

    def predict(self, X):

        X = check_array(X, dtype=np.float32, order="C", accept_sparse='csr')
        raw_predictions = self.loss_.get_init_raw_prediciton(X, self.init_)
        # print(raw_predictions)
        for estimator in self.estimators_:
            # estimator.predict(X)
            raw_predictions[:,0] += self.learning_rate * estimator[0].predict(X)            

        return raw_predictions