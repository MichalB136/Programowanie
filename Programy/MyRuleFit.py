import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from MyBoostingClassifier import MyGradientBoosting

class Winsorizer():
    
    def __init__(self, trim_quantile=0.0):
        self.trim_quantile = trim_quantile
        self.winsor_lims = None

    def train(self, X):
        self.winsor_lims = np.ones([2, X.shape[1]]) * np.inf
        self.winsor_lims[0, :] =- np.inf
        if self.trim_quantile > 0:
            for i_col in np.arange(X.shape[1]):
                lower = np.percentile(X[:, i_col], self.trim_quantile * 100)
                upper = np.percentile(X[:, i_col], 100 - self.trim_quantile * 100)
                self.winsor_lims[:, i_col] = [lower, upper]

    def trim(self, X):
        X_ = X.copy()
        X_ = np.where(X > self.winsor_lims[1,:], \
                        np.tile(self.winsor_lims[1 ,:],\
                                 [X.shape[0],1]), \
                        np.where(X < self.winsor_lims[0, :], 
                                 np.tile(self.winsor_lims[0, :],
                                         [X.shape[0],1]), X)) 
        return X_

class RuleEnsemble():
    def __init__(self, tree_list, feature_names=None):
        self.tree_list = tree_list
        self.feature_names = feature_names
        self.rules = set()
        ## TODO: Move this out of __init__
        self._exctract_rules()
        self.rules = list(self.rules)
    
    def extract_rules_from_tree(self):
        return rules

    def _extract_rules(self):
        for tree in self.tree_list:
            rules = extract_rules_from_tree(tree[0].tree_, 
                                            feature_names=self.feature_names)
            self.rules.update(rules)


class FriedScale():
    def __init__(self, winsorizer = None):
        self.scale_multipliers = None
        self.winsorizer = winsorizer

    def train(self, X):
        if self.winsorizer != None:
            X_trimmed = self.winsorizer.trim(X)
        else:
            X_trimmed = X

        scale_multipliers = np.ones(X.shape[1])
        for i_col in np.arange(X.shape[1]):
            num_uniq_vals = len(np.unique(X[:, i_col]))
            if num_uniq_vals > 2:
                scale_multipliers[i_col] = \
                    0.4 / (1.0e-12 + np.std(X_trimmed[:, i_col]))
        self.scale_multipliers = scale_multipliers

    def scale(self, X):
        if self.winsorizer != None:
            return self.winsorizer.trim(X) * self.scale_multipliers
        else:
            return X * self.scale_multipliers


class MyRuleFit(BaseEstimator, TransformerMixin):

    def __init__(self, tree_size=4, sample_fract='default',
                 max_rules=2000, memory_par=0.01, tree_generator=None,
                 rfmode='classify', lin_trim_quantile=0.025, 
                 lin_standardise=True, exp_rand_tree_size=True,
                 model_type='rl', Cs=None, cv=3, tol=0.0001,
                 max_iter=None, n_jobs=None, random_state=None):
        self.tree_generator = tree_generator
        self.rfmode = rfmode
        self.lin_trim_quantile = lin_trim_quantile
        self.lin_standardise = lin_standardise
        self.winsorizer = Winsorizer(trim_quantile=lin_trim_quantile)
        self.friedscale = FriedScale(self.winsorizer)
        self.stddev = None
        self.mean = None
        self.exp_rand_tree_size = exp_rand_tree_size
        self.max_rules = max_rules
        self.sample_fract = sample_fract
        self.max_rules = max_rules
        self.memory_par = memory_par
        self.tree_size = tree_size
        self.random_state = random_state
        self.model_type = model_type
        self.cv = cv
        self.tol = tol 
        self.max_iter = 1000 if 'regress' else 100
        self.n_jobs = n_jobs
        self.Cs = Cs
                 
    def fit(self, X, y=None, feature_names=None):

        n_samples = X.shape[0]
        if feature_names is None:
            self.feature_names = ['feature_' + str(x) for x in range(0, X.shape[1])]
        else:
            self.feature_names = feature_names
        if 'r' in self.model_type:
            if self.tree_generator is None:
                n_estimators_default = \
                    int(np.ceil(self.max_rules / self.tree_size))
                self.sample_fract = min(0.5, (100 + 6 * np.sqrt(n_samples)) 
                                                                / n_samples)
                if self.rfmode == 'classify':
                    self.tree_generator = \
                        MyGradientBoosting(n_estimators=n_estimators_default,
                                           max_leaf_nodes=self.tree_size,
                                           learning_rate=self.memory_par,
                                           random_state=self.random_state,
                                           max_depth=100)
                else:
                    raise ValueError('nie zrobilem regresora :)')

        self.tree_generator.fit(X, y)
        tree_list = self.tree_generator.estimators_
        
        self.rule_ensemble = RuleEnsemble(tree_list=tree_list,
                                          feature_names=self.feature_names)
                    
        return self