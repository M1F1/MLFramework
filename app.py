from sklearn import externals
import joblib
from settings import PROJECT_ROOT
import os
models_path = os.path.join(PROJECT_ROOT, 'models')
model_filename = 'best_model_10_03_2019_11_54_02.joblib'
model_path = os.path.join(models_path, model_filename)
clf = joblib.load(model_path)

def predict(input):
    prediction_value = clf.predict([input])[0]
    return {'prediction': probas}

