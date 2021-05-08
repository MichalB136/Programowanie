#%%
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import numpy as np
import pandas as pd
import psycopg2

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
conn = psycopg2.connect("dbname=mimic user=postgres password=Boychan!36 port=5433")
cur = conn.cursor()
# cur.execute("CREATE TABLE test (id serial PRIMARY KEY, num integer, data varchar);")
# cur.execute("INSERT INTO test (num, data) VALUES (%s, %s)",(100, "abc'def"))
#%%
cur.execute("""SELECT ie.subject_id, ie.hadm_id, ie.icustay_id,
    ie.intime, ie.outtime, adm.deathtime,
    ROUND((cast(ie.intime as date) - cast(pat.dob as date))/365.242, 2) AS age,
    ROUND((cast(ie.intime as date) - cast(adm.admittime as date))/365.242, 2) AS preiculos,
    CASE
        WHEN ROUND((cast(ie.intime as date) - cast(pat.dob as date))/365.242, 2) <= 1
            THEN 'neonate'
        WHEN ROUND((cast(ie.intime as date) - cast(pat.dob as date))/365.242, 2) <= 14
            THEN 'middle'
        -- all ages > 89 in the database were replaced with 300
        WHEN ROUND((cast(ie.intime as date) - cast(pat.dob as date))/365.242, 2) > 100
            THEN '>89'
        ELSE 'adult'
        END AS ICUSTAY_AGE_GROUP,
    -- note that there is already a "hospital_expire_flag" field in the admissions table which you could use
    CASE
        WHEN adm.hospital_expire_flag = 1 then 'Y'           
    ELSE 'N'
    END AS hospital_expire_flag,
    -- note also that hospital_expire_flag is equivalent to "Is adm.deathtime not null?"
    CASE
        WHEN adm.deathtime BETWEEN ie.intime and ie.outtime
            THEN 'Y'
        -- sometimes there are typographical errors in the death date, so check before intime
        WHEN adm.deathtime <= ie.intime
            THEN 'Y'
        WHEN adm.dischtime <= ie.outtime
            AND adm.discharge_location = 'DEAD/EXPIRED'
            THEN 'Y'
        ELSE 'N'
        END AS ICUSTAY_EXPIRE_FLAG
     FROM icustays ie
     INNER JOIN patients pat
     ON ie.subject_id = pat.subject_id
     INNER JOIN admissions adm
     ON ie.hadm_id = adm.hadm_id;
               """)
test = cur.fetchall()
# cur.close()
# conn.close()
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