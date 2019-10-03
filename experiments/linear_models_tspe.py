from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA

import pandas as pd
import pickle
import os
from settings import PROJECT_ROOT
from matplotlib import pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
import numpy as np
import pdb
from scipy.stats import uniform, randint
import xgboost as xgb

# set_logger
model_filename = 'best_model_{}.joblib'.format(datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))
logPath = os.path.join(PROJECT_ROOT, 'models')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.FileHandler("{0}/{1}.log".format(logPath, model_filename)),
        logging.StreamHandler()
    ])
logger = logging.getLogger()

# get_data
# filename = 'file_test.csv'
# data_path = os.path.join(PROJECT_ROOT, 'data', filename)
# df = pd.read_csv(data_path, index_col=0)
path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(path)
df = df.select_dtypes(include=np.number)
df = df.dropna()
print(df.head())

# choose independent features and indicator feature
# independent_column1, independent_column2, independent_column3, independent_column4 = "x1", "x2", "x3", "x4"
dependent_column = 'price'
x_data = df.drop(dependent_column, axis=1)
y_data = df[dependent_column]

# choose model type
# model = LinearRegression()
# model = Ridge()
# pca = PCA()
# model = LogisticRegression()
# model = SGDClassifier()
model = xgb.XGBRegressor(objective="reg:linear", random_state=42)
logger.info("Model type: {}".format(model.__class__.__name__))
# define pipeline
pipe = Pipeline([('scale', StandardScaler()),
                 # ('pca', pca),
                 # ('polynomial', PolynomialFeatures(include_bias=False)),
                 ('xgb', model),
                 # ('ridge', model)
                 ])

logger.info("Execute grid search:")
# pdb.set_trace()
size = 2
seed = 42
np.random.seed(seed=seed)
logger.info("random seed: {}".format(seed))
param_grid = {
              # 'ridge__alpha': [0.001, 0.1, 1, 10, 100, 1000, 10000, 100000, 100000],
              # 'polynomial__degree': [1, 2, 3],
              # 'pca__n_components': [2, 5, 6, 10, 12, 15],
              "xgb__colsample_bytree":uniform(0.7, 0.3).rvs(size=size) ,
              "xgb__gamma": uniform(0, 0.5).rvs(size=size),
              "xgb__learning_rate": uniform(0.03, 0.3).rvs(size), # default 0.1
              "xgb__max_depth": randint(2, 6).rvs(size), # default 3
              "xgb__n_estimators": randint(100, 150).rvs(size), # default 100
              "xgb__subsample": uniform(0.6, 0.4).rvs(size)
              }
logger.info("param grid: {}".format(param_grid))

search = GridSearchCV(pipe, param_grid, cv=4)
search.fit(x_data, y_data)
logger.info('best_score: {}'.format(search.best_score_))
logger.info('best_params: {}'.format(search.best_params_))
best_model = search.best_estimator_

logger.info('Holdout:')
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.20, random_state=1)
logger.info("number of test samples: {}".format(x_test.shape[0]))
logger.info("number of training samples: {}".format(x_train.shape[0]))
best_model.fit(x_train, y_train)
logger.info('in-sample evaluation')
logger.info('score: {}'.format(best_model.score(x_train, y_train)))
logger.info('out-sample evaluation')
logger.info('score: {}'.format(best_model.score(x_test, y_test)))

logger.info('Cross-validation:')
scores = cross_val_score(best_model, x_data, y_data, cv=4)
# pdb.set_trace()
logger.info("The mean of the folds are {} and the standard deviation is {}".format(scores.mean(), scores.std()))
from joblib import dump, load
model_path = os.path.join(PROJECT_ROOT, 'models', model_filename)
logger.info('serializing best model...')
dump(best_model, model_path)
logger.info('done.')
logger.info('sanity check: loading model from {}'.format(model_path))
loaded_model_sanity_check = load(model_path)
loaded_model_sanity_check.predict(x_test)
logger.info('done.')


if model.__class__.__name__ in ('XGBClassifier'):
    xgb.plot_importance(best_model)
    xgb.to_graphviz(best_model, num_trees=best_model.best_iteration)



