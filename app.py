from sklearn import externals
import joblib
model_filename = 'hatespeech.joblib.z'
clf = joblib.load(model_filename)

def predict(input):
    probas = clf.predict_proba([input])[0]
    return {'hate speech': probas[0],
           'offensive language': probas[1],
           'neither': probas[2]}