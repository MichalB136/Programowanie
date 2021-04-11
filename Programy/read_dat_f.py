#%%
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston, make_hastie_10_2, make_classification
from sklearn.metrics import mean_absolute_error
from rulefit import RuleFit
#%%
from MyBoostingClassifier import MyGradientBoosting, LeastSquareLossFunction
#%%
d = {'age': [5,11,14,8,12,10], 
     'location': [5,12,6,4,9,11], 
     'square_footage':[1500,2030,1442,2501,1300,1789], 
     'price':[480.0,1090.0,350.0,1310.0,400.0,500.0] 
     }
df = pd.DataFrame(data=d)
feature_names = ['age', 'location', 'square_footage']
#%%
X = pd.DataFrame(d, columns=feature_names)
y = pd.DataFrame(d, columns=['price'])
# X = X.to_numpy(dtype=np.float32)
# y = y.to_numpy(dtype=np.float64)
#%%
regresor = MyGradientBoosting()
test = regresor.fit(X, y)
#%%
regresor2 = GradientBoostingRegressor( n_estimators=10)
test2 = regresor2.fit(X, y)
# test = regresor.pseudo_res(1, y)
# regresor.fit_stage(1, X, y, 5, None, None, None)
# regresor.compute(1, y)

#%%
# boston = load_boston()
# X = pd.DataFrame(boston.data, columns=boston.feature_names)
# y = pd.Series(boston.target)

# X_train, X_test, y_train, y_test = train_test_split(X, y)

# regressor = GradientBoostingRegressor(
#     max_depth=2,
#     n_estimators=2,
#     learning_rate=1.0
# )
# regressor.fit(X_train, y_train)
# y_pred = regressor.predict(X_test)
# mean_absolute_error(y_test, y_pred)
# X, y = make_classification(random_state=0)
# X_train, X_test, y_train, y_test = train_test_split(
#      X, y, random_state=0)
# clf = GradientBoostingClassifier(random_state=0)
# print(X_train)
# print(y_train)
# clf.fit(X_train, y_train)
# clf.predict(X_test[:2])
# clf.score(X_test, y_test)
