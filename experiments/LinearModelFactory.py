from scipy.stats import uniform, randint
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn import svm
import xgboost as xgb


class LinearModelFactory:
    def __init__(self, h_param_n: int, seed: int):
        self.h_param_n = h_param_n
        self.seed = seed

    def get_model_and_param_grid(self, model_name: str):
        if model_name == 'ridge':
            return self.get_ridge()
        elif model_name == 'xgboost_regressor':
            return self.get_xgboost_regressor()
        elif model_name == 'xgboost_classifier':
            return self.get_xgboost_classifier()
        elif model_name == 'sgd_classifier':
            return self.get_sgd_classifier()
        elif model_name == 'svm':
            return self.get_svm_classifier()
        else:
            raise ValueError(model_name)

    def get_sgd_classifier(self):
        model = SGDClassifier(max_iter=1000, tol=1e-3)
        param_grid = {'model__alpha': [0.001]}
        return model, param_grid

    def get_svm_classifier(self):
        model = svm.SVC(gamma='scale')
        param_grid = {'model__kernel': ['rbf'],
                      'model__degree': [5],
                     }
        return model, param_grid

    def get_ridge(self):
        model = Ridge()
        param_grid = {'model__alpha': [0.001, 0.1, 1, 10, 100, 1000, 10000, 100000, 100000]}
        return model, param_grid

    def get_xgboost_regressor(self):
        model = xgb.XGBRegressor(objective="reg:squarederror", random_state=self.seed)
        param_grid = {
                      "model__colsample_bytree": uniform(0.7, 0.3).rvs(size=self.h_param_n),
                      "model__gamma": uniform(0, 0.5).rvs(size=self.h_param_n),
                      "model__learning_rate": uniform(0.03, 0.3).rvs(size=self.h_param_n),  # default 0.1
                      "model__max_depth": randint(2, 6).rvs(size=self.h_param_n),  # default 3
                      "model__n_estimators": randint(100, 150).rvs(size=self.h_param_n),  # default 100
                      "model__subsample": uniform(0.6, 0.4).rvs(size=self.h_param_n)
                     }

        return model, param_grid

    def get_xgboost_classifier(self,):
        model = xgb.XGBClassifier(objective='binary:logistic', seed=self.seed)
        param_grid = {
                      "model__colsample_bytree": uniform(0.7, 0.3).rvs(size=self.h_param_n),
                      "model__max_depth": randint(2, 3).rvs(size=self.h_param_n),  # default 3
                      "model__learning_rate": uniform(0.03, 0.3).rvs(size=self.h_param_n),  # default 0.1
                      "model__gamma": uniform(0, 0.5).rvs(size=self.h_param_n),
                      "model__n_estimators": randint(25, 30).rvs(size=self.h_param_n),  # default 100
                      "model__subsample": uniform(0.6, 0.4).rvs(size=self.h_param_n)
                     }
        return model, param_grid

