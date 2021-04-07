import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor

class MeanSquareLossFunction():

    def __init__(self, n_classes):
        self.K = n_classes

    def __call__(self, y, raw_predictions, sample_weight=None):
        raw_predictions = raw_predictions.ravel()
        if sample_weight is None:
            return -1 * sum((y - raw_predictions))

    def init_estimator(self):
        return DummyRegressor(strategy='mean')

    def get_init_raw_prediciton(self, X, estimator):
        predictions = estimator.predict(X)
        return predictions.reshape(-1,1).astype(np.float64)



class MyGradientBoosting():

    def __init__(self, loss, learning_rate, max_depth, 
                 n_estimators, n_classes, criterion, min_samples_split,
                 init, min_samples_leaf, min_weight_fraction_leaf,
                 min_impurity_decrease, min_impurity_split, max_features, 
                 random_state, ccp_alpha, max_lead_nodes=None):
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
        self.init = init        

    def _initialize(self, X, y):
        # inicjalizacja procesu
        if self.loss == 'square':
            loss_class = MeanSquareLossFunction
            self.loss_ = loss_class(self.n_classes)
        self.init_ = self.init
        if self.init_ is None:
            self.init_ = self.loss_.init_estimator()
        self.estimators_ = np.empty((self.n_estimators, self.loss_.K), dtype=np.object)
        self.residual_ = np.empty((self.n_estimators, y.shape[0]), dtype=np.float64)
        self.gamma_ = np.zeros((self.n_estimators, 1), dtype=np.float64)
        self.estimators = np.zeros(X.shape[0])
        self._mean_y = np.mean(y)

    def pseudo_res(self, i, y):
        p_val = self.estimators[i - 1]
        #Compute pseudo-residual
        residual = [-(self.loss_(y_val, p_val)) for y_val in y] 
        residual = np.asarray(residual, dtype=np.float64)
        return residual

    def _fit_stage(self, i, X, y, max_depth,
                   sample_weight, sample_mask, random_state):
        #Fit weka learner 
        self.residual_[i] = self.pseudo_res(i, y)

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
        
        tree.fit(X, self.residual_[i], sample_weight=sample_weight, 
                 check_input=False)

        self.estimators_[i] = tree

    def compute(self, i, y):
        for k, val_k in enumerate(y):
            self.gamma_[i] =+ -2 * self.residual_[i][k] * \
            (self.residual_[i][k] * self._mean_y + val_k - self.estimators[i - 1])
        # self.gamma_[i] = sum(self.residual_[i] * \
        #     (self.residual_[i] * self._mean_y + y - self.estimators[i - 1]))
        #sprawdz rownanie 
    
    # def update_model(self, i, learning_rate):
    #     self.estimators[i] = self.estimators[i - 1] + learning_rate * \
    #         self.gamma_
    def _is_initialized(self):
        return len(getattr(self, 'estimators_', [])) > 0 

    def fit(self, X, y, sample_weight=None):

        if not self._is_initialized():

            self._initialize(X, y)
            if self.init_ == 'zero':
                raw_predicitons = np.zeros(shape=(X.shape[0], self.loss_.K),
                                        dtype=np.float64)
            else:
                if sample_weight == None:
                    self.init_.fit(X,y)

                raw_predicitons = self.loss_.get_init_raw_prediciton(X, self.init_)

        #fit stage funckja?
        
        #return raw_predicitons wyplute na konic calego algorytmu?