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
from MyRuleFit import MyRuleFit

#%%
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

y2 = y.iloc[0:100]
X2 = X.iloc[0:100]
X_train, X_test, y_train, y_test = train_test_split(X2, y2)
#%%
classifier = MyGradientBoosting(n_estimators=100)
# classifier = GradientBoostingClassifier( n_estimators=10)
rf = MyRuleFit(tree_generator=classifier)
# rf = RuleFit()
test = rf.fit(X_train, y_train, iris.feature_names)
#%%
y_predict = test.predict(X_test)

#%% 
rules = test.get_rules()
#%%
regresor2 = GradientBoostingClassifier( n_estimators=10)
test2 = regresor2.fit(X2, y2)
# test = regresor.pseudo_res(1, y)
# regresor.fit_stage(1, X, y, 5, None, None, None)
# regresor.compute(1, y)



#%%
boston_data = pd.read_csv("boston.csv", index_col=0)

y = boston_data.medv.values
X = boston_data.drop("medv", axis=1)
features = X.columns
X = X.as_matrix()

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