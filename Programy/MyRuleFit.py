import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from MyBoostingClassifier import MyGradientBoosting
from sklearn.linear_model import LassoCV,LogisticRegressionCV
from functools import reduce
class RuleCondition():

    def __init__(self, feature_index, threshold, operator, 
                 support, feature_name=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.operator = operator
        self.support = support
        self.feature_name = feature_name

    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        if self.feature_name:
            feature = self.feature_name
        else:
            feature = self.feature_index
        return f"{feature} {self.operator} {self.threshold}"
    
    def transform(self, X):
        if self.operator == "<=":
            res = 1 * (X[:, self.feature_index] <= self.threshold)
        elif self.operator == ">":
            res = 1 * (X[:, self.feature_index] > self.threshold)
        return res

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()
    
    def __hash__(self):
        return hash((self.feature_index, self.threshold, self.operator,
                     self.feature_name))

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

class Rule():

    def __init__(self, rule_conditions, prediction_value):
        self.conditions = set(rule_conditions)
        self.support = min([x.support for x in rule_conditions])
        self.prediciton_value = prediction_value
        self.rule_direction = None

    def transform(self, X):

        rule_applies = [condition.transform(X) for condition in self.conditions]
        return reduce(lambda x,y: x * y, rule_applies)

    def __str__(self):
        return " & ".join([x.__str__() for x in self.conditions])

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return sum([condition.__hash__() for condition in self.conditions])

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

class RuleEnsemble():
    def __init__(self, tree_list, feature_names=None):
        self.tree_list = tree_list
        self.feature_names = feature_names
        self.rules = set()
    
    def extract_rules_from_tree(self, tree, feature_names=None):
        rules = set()
        self.traverse_nodes(tree, feature_names, rules)
        return rules

    def traverse_nodes(self, tree, feature_names, rules, node_id=0, operator=None, threshold=None,
                    feature=None, conditions=[]):
        if node_id != 0:
            if feature_names is not None:
                feature_name = feature_names[feature]
            else:
                feature_name = feature
            rule_condtion = RuleCondition(feature, threshold, operator, 
                                          tree.n_node_samples[node_id] 
                                          / float(tree.n_nodes_samples[0]),
                                          feature_name)
            new_conditions = conditions + [rule_condtion]
        else:
            new_conditions = []

        if tree.childer_left[node_id] != tree.children_right[node_id]:
            feature = tree.feature[node_id]
            threshold = tree.threshold[node_id]

            left_node_id = tree.children_left[node_id]
            self.traverse_nodes(left_node_id, "<=", threshold, 
                                feature, new_conditions)

            right_node_id = tree.children_right[node_id]
            self.traverse_nodes(right_node_id, ">", threshold, 
                                feature, new_conditions)
        else:
            if len(new_conditions) > 0:
                new_rule = Rule(new_conditions, tree.value[node_id][0][0])
                rules.update([new_rule])
            else:
                pass
            return None


    def _extract_rules(self, tree_list, feature_names=None):
        for tree in tree_list:
            rules = self.extract_rules_from_tree(tree[0].tree_, 
                                            feature_names=self.feature_names)
            self.rules.update(rules)
            self.rules = list(self.rules)

    def transform(self, X, coefs=None):

        self._extract_rules(self.tree_list, self.feature_names)

        rule_list = list(self.rules) ## TODO: sprawdzic czy to potrzebne

        if coefs is None:
            return np.array([rule.transform(X) for rule in rule_list]).T
        else:
            res = np.array([rule_list[i_rule].transform(X) for i_rule \
                  in np.arange(len(rule_list)) if coefs[i_rule] != 0]).T
            res_ = np.zeros([X.shape(0), len(rule_list)])
            res_[:, coefs != 0] = res
            return res_
        
    def __str__(self):
        return (map(lambda x: x.__str__(), self.rules)).__str__()




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
            
            X_rules = self.rule_ensemble.transform(X)

        if 'l' in self.model_type:
            
            self.winsorizer.train(X)
            winsorized_X = self.winsorizer.trim(X)
            self.stddev = np.std(winsorized_X, axis=0)
            self.mean = np.mean(winsorized_X, axis=0)

            if self.lin_standardise:
                self.friedscale.train(X)
                X_regn = self.friedscale.scale(X)
            else:
                X_regn = X.copy()

        ## Complie training data
        X_concat = np.zeros([X.shape[0], 0])
        if 'l' in self.model_type:
            X_concat = np.concatenate((X_concat, X_regn), axis=1)
        if 'r' in self.model_type:
            if X_rules.shape[0] > 0:
                X_concat = np.concatenate((X_concat, X_rules), axis=1)

        # fit lasso
        if self.rfmode == 'regress':
            if self.Cs is None:
                n_alphas = 100
                alphas = None
            elif hasattr(self.Cs, "__len__"):
                n_alphas = None
                alphas = 1. / self.Cs
            else:
                n_alphas = self.Cs
                alphas = None
            self.lscv = LassoCV(n_alphas=n_alphas, alphas=alphas, cv=self.cv,
                                max_iter=self.max_iter, tol=self.tol,
                                n_jobs=self.n_jobs, random_state=self.random_state)
            self.lscv.fit(X_concat, y)
            self.coef_ = self.lscv.coef_
            self.intercept_ = self.lscv.intercept_
        else:
            Cs = 10 if self.Cs is None else self.Cs
            self.lscv = LogisticRegressionCV(Cs=Cs, cv=self.cv, penalty='l1',
                                             max_iter=self.max_iter,
                                             tol=self.tol, n_jobs=self.n_jobs,
                                             random_state=self.random_state,
                                             solver='liblinear')
            self.lscv.fit(X_concat, y)
            self.coef_ = self.lscv.coef_[0]
            self.intercept_ = self.lscv.intercept_[0]
                    
        return self