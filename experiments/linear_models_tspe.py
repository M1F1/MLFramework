from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from category_encoders import OrdinalEncoder
from category_encoders import OneHotEncoder

import matplotlib.pyplot as plt

import pandas as pd
import os
from settings import PROJECT_ROOT
import logging
from datetime import datetime
import numpy as np
import pdb
import xgboost as xgb
from experiments.LinearModelFactory import LinearModelFactory
from joblib import dump, load

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

# get local data

filename = 'male_data_0.csv'
data_path = os.path.join(PROJECT_ROOT, 'data')
if not os.path.exists(data_path):
    os.makedirs(data_path)
data_filename_path = os.path.join(data_path, filename)
df = pd.read_csv(data_filename_path, index_col=0)
print(df.head())
# get remote data
# path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
# df = pd.read_csv(path)
# df = df.dropna()
# print(df.head())

# choose independent features and indicator
# be careful about information leak
dependent_column = 'is_obesity'
y_data = df[dependent_column]
print(y_data.head())
x_data = df.drop(columns=[dependent_column, 'log_BMI', 'IID'], axis=1)#.select_dtypes(include=np.number)
print(x_data.head())
print(x_data.dtypes)

# TODO: add clustering model
# TODO: deploy on production
# TODO: add SQL handlers aka ORM
# TODO: ADD preparing data, in this file just load it

seed = 42
folds_num = 4
hyperparameters_num = 2
np.random.seed(seed=seed)
# choose model type, look at LinearModelFactory to choose one
model_name = 'xgboost_classifier'

model, param_grid = LinearModelFactory(seed=seed, h_param_n=hyperparameters_num)\
                    .get_model_and_param_grid(model_name)
logger.info("Model type: {}".format(model.__class__.__name__))
# define pipeline
# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))])
#
categorical_features = x_data.select_dtypes(include=object).columns.values
preprocessor = ColumnTransformer(transformers=[
                                # ('one_hot', OneHotEncoder(), categorical_features),
                                ('encoder', OrdinalEncoder(), categorical_features),
                                ])
# print('categorical_features:', categorical_features)
pipe = Pipeline([
                 # ('scale', StandardScaler()),
                 ('preprocessor', preprocessor),
                 # ('encoder', OneHotEncoder(cols=categorical_features)),
                 # ('encoder', OrdinalEncoder(cols=categorical_features)),
                 # ('pca', PCA()),
                 # ('polynomial', PolynomialFeatures(include_bias=False)),
                 ('model', model)
                 ])
# param_grid['polynomial__degree'] = [1, 2, 3]
# param_grid['pca__n_components'] = [2, 5, 6, 10, 12, 15]

logger.info("Execute grid search:")
logger.info("random seed: {}".format(seed))
logger.info("param grid: {}".format(param_grid))
search = GridSearchCV(pipe, param_grid, cv=folds_num)
search.fit(x_data, y_data)
logger.info('best_score: {}'.format(search.best_score_))
logger.info('best_params: {}'.format(search.best_params_))
best_model = search.best_estimator_

logger.info('Holdout:')
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.20, random_state=seed)
logger.info("number of test samples: {}".format(x_test.shape[0]))
logger.info("number of training samples: {}".format(x_train.shape[0]))
best_model.fit(x_train, y_train)
logger.info('in-sample evaluation')
logger.info('score: {}'.format(best_model.score(x_train, y_train)))
logger.info('out-sample evaluation')
logger.info('score: {}'.format(best_model.score(x_test, y_test)))

logger.info('Cross-validation:')
scores = cross_val_score(best_model, x_data, y_data, cv=folds_num)
logger.info("The mean of the folds are {} and the standard deviation is {}".format(scores.mean(), scores.std()))

logger.info('Serializing best model...')
models_path = os.path.join(PROJECT_ROOT, 'models')
if not os.path.exists(models_path):
    os.makedirs(models_path)
model_path = os.path.join(models_path, model_filename)
dump(best_model, model_path)
logger.info('done.')
logger.info('Sanity check -> loading model from {} ...'.format(model_path))
loaded_model_sanity_check = load(model_path)
loaded_model_sanity_check.predict(x_test)
logger.info('done.')

# visualizaton based on model name
if model.__class__.__name__ in ('XGBClassifier'):
    cols_names = x_data.columns.values.tolist()
    xgb_model = best_model.named_steps["model"]
    xgb_model.get_booster().feature_names = cols_names
    xgb.plot_importance(xgb_model)

    vis_path = os.path.join(PROJECT_ROOT, 'visualizations')
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    fig = plt.gcf()
    # fig.set_size_inches(75, 50)
    filename_path = os.path.join(vis_path,
                                 'feature_importance_{}.png'.format(datetime.now().strftime("%m_%d_%Y_%H_%M_%S")))
    fig.savefig(filename_path)
    plt.show()

    # need to install: sudo apt-get install graphviz
    xgb.plot_tree(xgb_model, num_trees=xgb_model.get_booster().best_iteration)
    fig = plt.gcf()
    fig.set_size_inches(75, 50)
    filename_path = os.path.join(vis_path, 'tree_{}.png'.format(datetime.now().strftime("%m_%d_%Y_%H_%M_%S")))
    fig.savefig(filename_path)
    plt.show()



