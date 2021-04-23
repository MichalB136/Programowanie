#%%
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import numpy as np
import pandas as pd

from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import issparse
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston, make_hastie_10_2, make_classification, load_iris
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree._tree import TREE_LEAF
from sklearn.utils import check_random_state
from sklearn.utils.validation import _check_sample_weight
from rulefit import RuleFit
#%%
from MyBoostingClassifier import MyGradientBoosting
from MyLossFunction import BinomialDeviance
#%%
d = {'age': [5,11,14,8,12,10], 
     'location': [5,12,6,4,9,11], 
     'square_footage':[1500,2030,1442,2501,1300,1789], 
     'price':[480.0,1090.0,350.0,1310.0,400.0,500.0] 
     }
df = pd.DataFrame(data=d)
feature_names = ['age', 'location', 'square_footage']

d_test = {'age': [6,3,15,10,8,11], 
     'location': [3,6,11,5,12,14], 
     'square_footage':[2000,1000,1342,1501,1255,1189], 
     'price':[880.0,480.0,250.0,1110.0,2000.0,600.0] 
     }
df_test = pd.DataFrame(data=d_test)
feature_names = ['age', 'location', 'square_footage']
#%%
X = pd.DataFrame(d, columns=feature_names)
y = pd.DataFrame(d, columns=['price'])
X_test = pd.DataFrame(d_test, columns=feature_names)
y_test = pd.DataFrame(d_test, columns=['price'])
# X = X.to_numpy(dtype=np.float32)
# y = y.to_numpy(dtype=np.float64)
#%%
regresor = MyGradientBoosting(n_estimators=100)
test = regresor.fit(X2, y2)
y_predict = test.predict(X2)

#%%
regresor2 = GradientBoostingClassifier( n_estimators=10)
test2 = regresor2.fit(X2, y2)
# test = regresor.pseudo_res(1, y)
# regresor.fit_stage(1, X, y, 5, None, None, None)
# regresor.compute(1, y)


#%%
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)
X_train, X_test, y_train, y_test = train_test_split(X, y)
y2 = y.iloc[0:100]
X2 = X.iloc[0:100]
#%%
X2 = X2.values
y2 = y2.values

X2 = np.array(X2, dtype=np.float32)
# for i, t in enumerate(y):
#      if t == 2:
#           print(i)
#%%
criterion='friedman_mse'
max_depth=3
min_samples_split=2
min_samples_leaf=1
min_weight_fraction_leaf=0
min_impurity_decrease=0
min_impurity_split=None
max_features=None
max_lead_nodes=None
random_state = check_random_state(None)
ccp_alpha=0.0
max_leaf_nodes=None
X_idx_sorted = None
sample_weight = None
learning_rate = 0.1

sample_weight = _check_sample_weight(sample_weight, X2)

n_samples = X2.shape[0]
sample_mask = np.ones((n_samples, ), dtype=np.bool)


n_classes_ = 2
loss_class = BinomialDeviance
loss = loss_class(n_classes_)
init_ = loss.init_estimator()
init_.fit(X2, y2)
raw_predictions = loss.get_init_raw_prediciton(X2, init_)
print(raw_predictions.shape)
raw_predictions_copy = raw_predictions.copy()
original_y = y2
X_csr = csr_matrix(X2) if issparse(X2) else None
#%%
for k in range(loss.K):
     if loss.is_multi_class:
          y2 = np.array(original_y == k, dtype=np.float64)

     residual = loss.negative_gradient(y2, raw_predictions_copy, k=k,
                                                  sample_weight=sample_weight)
     #2.2
     tree = DecisionTreeRegressor(
          criterion=criterion,
          splitter='best',
          max_depth=max_depth,
          min_samples_split=min_samples_split,
          min_samples_leaf=min_samples_leaf,
          min_weight_fraction_leaf=min_weight_fraction_leaf,
          min_impurity_decrease=min_impurity_decrease,
          min_impurity_split=min_impurity_split,
          max_features=max_features,
          max_leaf_nodes=max_leaf_nodes,
          random_state=random_state,
          ccp_alpha=ccp_alpha)
     X2 = X_csr if X_csr is not None else X2
     tree.fit(X2, residual, sample_weight=sample_weight,
               check_input=False, X_idx_sorted=X_idx_sorted)
     #2.4
     print(k)
     loss.update_terminal_regions(
          tree.tree_, X2, y2, residual, raw_predictions, sample_weight,
          sample_mask, learning_rate=learning_rate, k=k)    
     print('stop')  
# print(raw_predictions)


#%%
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target)

X_train, X_test, y_train, y_test = train_test_split(X, y)

# regressor = GradientBoostingRegressor(
#     max_depth=2,
#     n_estimators=2,
#     learning_rate=1.0
# )
regresor.fit(X_train, y_train)
y_pred = regresor.predict(X_test)
mean_absolute_error(y_test, y_pred)
# X, y = make_classification(random_state=0)
# X_train, X_test, y_train, y_test = train_test_split(
#      X, y, random_state=0)
# clf = GradientBoostingClassifier(random_state=0)
# print(X_train)
# print(y_train)
# clf.fit(X_train, y_train)
# clf.predict(X_test[:2])
# clf.score(X_test, y_test)

# %%
