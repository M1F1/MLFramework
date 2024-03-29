FROM python:3.6
RUN pip install scikit-learn==0.20.2  firefly-python==0.1.15
COPY app.py model.joblib.z ./
CMD firefly app.predict --bind 0.0.0.0:5000
EXPOSE 5000