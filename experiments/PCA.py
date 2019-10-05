from utils import random_color
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler
from category_encoders import OrdinalEncoder
from utils import random_color
import pandas as pd
import numpy as np
import plotly
import credentials
plotly.tools.set_credentials_file(username='Qb1t', api_key=credentials.plotly_api_key)
import plotly.plotly as py
from datetime import datetime
from settings import PROJECT_ROOT
import os
from sklearn.compose import ColumnTransformer

# get data
# path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
# df = pd.read_csv(path)
# df = df.dropna()
# dependent_column = 'body-style'
# Y = df[dependent_column].values
# X = df.drop(dependent_column, axis=1).select_dtypes(include=np.number).values

filename = 'male_data_0.csv'
data_path = os.path.join(PROJECT_ROOT, 'data')
data_filename_path = os.path.join(data_path, filename)
df = pd.read_csv(data_filename_path, index_col=0)
dependent_column = 'is_obesity'
Y = df[dependent_column].replace([0, 1], ['normal', 'obesity'])
X = df.drop(columns=[dependent_column, 'log_BMI', 'IID'], axis=1)#.select_dtypes(include=np.number)

sklearn_pca = sklearnPCA(n_components=2)
# X_std = StandardScaler().fit_transform(X)
categorical_features = X.select_dtypes(include=object).columns.values
preprocessor = ColumnTransformer(transformers=[
    # ('one_hot', OneHotEncoder(), categorical_features),
    ('encoder', OrdinalEncoder(), categorical_features),
])
X_cat = preprocessor.fit_transform(X, Y)
print(X_cat)
Y_sklearn = sklearn_pca.fit_transform(X_cat)
print(Y_sklearn)
categories = np.unique(Y)
print(categories)
colors = dict()
for category in categories:
    colors[category] = random_color()
data = []

for category, color in colors.items():

    trace = dict(
        type='scatter',
        x=Y_sklearn[Y == category, 0],
        y=Y_sklearn[Y == category, 1],
        mode='markers',
        name=category,
        marker=dict(
            color=color,
            size=12,
            line=dict(
                color='rgba(217, 217, 217, 0.14)',
                width=0.5),
            opacity=0.8)
    )
    data.append(trace)

layout = dict(
        xaxis=dict(title='PC1', showline=False),
        yaxis=dict(title='PC2', showline=False)
)
fig = dict(data=data, layout=layout)
filename_path = os.path.join(PROJECT_ROOT, 'visualizations', 'pca-scikit-learn={}.html'.format(datetime.now().strftime("%m_%d_%Y_%H_%M_%S")))
plotly.offline.plot(fig, filename=filename_path)
py.iplot(fig, filename='pca-scikitlearn')
