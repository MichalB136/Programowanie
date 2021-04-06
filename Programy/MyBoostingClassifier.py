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

    def __init__(self, loss, learning_rate, max_depth, 
                 n_estimators, n_classes, criterion, min_samples_split, 
                 min_samples_leaf, min_weight_fraction_leaf, min_impurity_decrease,
                 min_impurity_split, max_features, random_state, ccp_alpha,
                 max_lead_nodes=None):
        self.learnig_rate = learning_rate
        self.loss = loss
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.n_classes = n_classes
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

    def get_loss_func(self):
        if self.loss == 'square':
            loss_class = SquareLossFunction
            self.loss_ = loss_class(self.n_classes)

    def initialize(self, X, y):
        # inicjalizacja procesu
        self.get_loss_func()
        self.estimators_ = np.empty((self.n_estimators, 1), dtype=np.object)
        self.estimators = np.zeros(X.shape[0])
        self.estimators[0] = self.loss_.get_init_raw_prediciton(y)

    def pseudo_res(self, i, y):
        p_val = self.estimators[i - 1]
        #Compute pseudo-residual
        residual = [-(self.loss_(y_val, p_val)) for y_val in y] 
        residual = np.asarray(residual, dtype=np.float64)
        return residual

    def _fit_stage(self, i, X, y, learning_rate, max_depth,
                   sample_weight, sample_mask, random_state):
        #Fit weka learner 
        residual = self.pseudo_res(i, y)

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
        
        tree.fit(X, residual, sample_weight=sample_weight, 
                 check_input=False)

        self.estimators_[i] = tree


        