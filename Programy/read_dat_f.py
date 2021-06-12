#%%
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import numpy as np
import pandas as pd
import csv
from psycopg2 import sql, connect
from psycopg2.extras import RealDictCursor, DictCursor
import matplotlib.pyplot as plt

from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import issparse
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.datasets import load_boston, make_hastie_10_2, make_classification, load_iris
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.tree._tree import TREE_LEAF
from sklearn.utils import check_random_state
from sklearn.utils.validation import _check_sample_weight
from rulefit import RuleFit
#%%
from MyBoostingClassifier import MyGradientBoosting
from MyLossFunction import BinomialDeviance
from MyRuleFit import MyRuleFit

#%%
conn = connect("dbname=mimic user=postgres password=Boychan!36 port=5433"
                         )
cur = conn.cursor()
target_cur = conn.cursor()

sql_string = """SELECT
	scnd.age,
	scnd.admission_type,
	scnd.renal,
	scnd.urea,
	scnd.pf,
	scnd.gcs,
	base.br,
	base.hr,
	base.sbp,
	base.wbc,
	base.temperature,
	base.k,
	base.na
--	base.dead
FROM my_tables.fist_table as base 
JOIN my_tables.second_table as scnd ON base.hadm_id = scnd.hadm_id
WHERE pf IS NOT NULL AND sbp IS NOT NULL AND temperature IS NOT NULL
LIMIT 680000
               """

target_string = """SELECT
	base.dead
FROM my_tables.fist_table as base 
JOIN my_tables.second_table as scnd ON base.hadm_id = scnd.hadm_id
WHERE pf IS NOT NULL AND sbp IS NOT NULL AND temperature IS NOT NULL
LIMIT 680000"""
# print(base_string)
cur.execute(sql_string)
base_table = cur.fetchall()
cur.close()
target_cur.execute(target_string)
target_table = target_cur.fetchall()
target_table = [item for t in target_table for item in t]
target_cur.close()
feature_names = ['age','admission_type','renal','urea', 'pf', 'gcs','br','hr','sbp','wbc','temperature','k','na']
#%%
X = pd.DataFrame(base_table, columns=feature_names)
y = pd.Series(target_table)
# X = X.to_numpy()
# y = y.to_numpy()
# y2 = y.iloc[0:100]
# X2 = X.iloc[0:100]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)

#%%
classifier = MyGradientBoosting(n_estimators=100, max_depth=5)
# classifier = GradientBoostingClassifier( n_estimators=10)
rf = MyRuleFit(tree_generator=classifier, model_type='rl')
# rf = RuleFit()
#%%
test = rf.fit(X_train, y_train, feature_names)
#%%
y_predict = test.predict(X_test)
#%%  to kie
rules = test.get_rules(exclude_zero_coef=True)
#%%
importance = test.get_feature_importance(exclude_zero_coef=True)
#%%
fig, axs = plt.subplots(figsize=(6, 4))
importance.plot(ax=axs,kind='bar', x='feature', y='importance')
axs.set_ylabel("Waga")
axs.set_xlabel("Cecha")
fig.savefig("test.png",facecolor='w', edgecolor='w',bbox_inches='tight')
#%%
mean_squared_error(y_test, y_predict)
#%%
accuracy_score(y_test, y_predict)
#%%