from flask import Flask, request
from gevent.pywsgi import WSGIServer

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def hello():
    sex = request.values.get('sex')
    model = request.values.get('model')
    if not model or not sex:
        raise Exception('Param undefined!')
    print('Request {}'.format(request.values))
    return 'OK'


if __name__ == '__main__':
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
