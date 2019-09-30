from sklearn.linear_model import LinearRegression
import pandas as pd
import os
from settings import PROJECT_ROOT
from matplotlib import pyplot as plt
import seaborn as sns

filename = 'file_test.csv'
data_path = os.path.join(PROJECT_ROOT, 'data', filename)
df = pd.read_csv(data_path, index_col=0)
independent_column1, independent_column2, independent_column3, independent_column4 = "x1", "x2", "x3", "x4"
dependent_column = 'y'
z = df[[independent_column1, independent_column2, independent_column3, independent_column4]]
y = df[dependent_column]

# Multiple Linear Regression
model = LinearRegression()
model.fit(z, y)

# polynomial features
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler

# multiple linear regression with polynomial features
# input = [('scale', StandardScaler()),
#          ('polynomial', PolynomialFeatures(include_bias=False)),
#          ('model', LinearRegression())]
# model= Pipeline(input)
# model.fit(z, y)

# IN-SAMPLE EVALUATION
y_hat = model.predict(z)
# R^2

from sklearn.metrics import r2_score
print('The R-square is: ', r2_score(y, y_hat))
from sklearn.metrics import mean_squared_error
print('The mean square error: ', mean_squared_error(y, y_hat))

# distribution plot
width = 12
height = 10
plt.figure(figsize=(width, height))
ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(y_hat, hist=False, color="b", label="Fitted Values", ax=ax1)
plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')
plt.show()
plt.close()
