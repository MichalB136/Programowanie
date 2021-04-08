import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor

class MeanSquareLossFunction():

    def __init__(self, n_classes):
        self.K = n_classes

    def __call__(self, y, raw_predictions, sample_weight=None):
        raw_predictions = raw_predictions.ravel()
        if sample_weight is None:
            return np.mean((y - raw_predictions.ravel()) ** 2)
        else:
            return (1 / sample_weight.sum() * np.sum(
                sample_weight * ((y - raw_predictions.ravel()) ** 2)))

    def init_estimator(self):
        return DummyRegressor(strategy='mean')

    def get_init_raw_prediciton(self, X, estimator):
        predictions = estimator.predict(X)
        return predictions.reshape(-1,1).astype(np.float64)

    def negative_gradient(self, y, raw_prediction):
        return y - raw_prediction


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

    # def fit_stage(self, X, y, max_depth,
    #                sample_weight, sample_mask, random_state):
    #     #Fit weka learner 
    #     self.residual_[i] = self.pseudo_res(i, y)

    #     tree = DecisionTreeRegressor(
    #         criterion=self.criterion,
    #         splitter='best',
    #         max_depth=self.max_depth,
    #         min_samples_split=self.min_samples_split,
    #         min_samples_leaf=self.min_samples_leaf,
    #         min_weight_fraction_leaf=self.min_weight_fraction_leaf,
    #         min_impurity_decrease=self.min_impurity_decrease,
    #         min_impurity_split=self.min_impurity_split,
    #         max_features=self.max_features,
    #         max_leaf_nodes=self.max_leaf_nodes,
    #         random_state=random_state,
    #         ccp_alpha=self.ccp_alpha)
        
    #     tree.fit(X, self.residual_[i], sample_weight=sample_weight, 
    #              check_input=False)

    #     self.estimators_ = tree
    
    def fit_stages(self, X, y, raw_predictions, sample_weight, random_state):
        
        residuals = self.loss_.negative_gradient(y, raw_predictions)
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
        
        tree.fit(X, residuals, sample_weight=sample_weight, check_input=False)

        h_m = tree.predict(X)
        # -hm(x) ( -gamma*hm(x) +yi - Fm-1(x))
        gamma = -h_m * (-raw_predictions * h_m + y - raw_predictions)
        # print(gamma)
        # print(f'Suma:{sum(gamma)}')
        self.estimators_ = tree
        return residuals
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
                print(raw_predicitons)


        #fit stage funckja?
        n_stages = self.fit_stages(X, y, raw_predicitons, random_state=None, sample_weight=None)

        return n_stages
        #return raw_predicitons wyplute na konic calego algorytmu?