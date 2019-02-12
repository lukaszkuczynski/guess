from flask import Flask, render_template, request, redirect
import datetime, logging
from logging import handlers
from predict import Predictor, from_pickle
import os, requests


app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.debug = True

LOG_FILENAME = 'app_access_logs.log'

app.logger.setLevel(logging.INFO) 

handler = handlers.RotatingFileHandler(
    LOG_FILENAME,
    maxBytes=1024 * 1024 * 100,
    backupCount=20
    )

app.logger.addHandler(handler)
app.logger.addHandler(logging.StreamHandler())

def download_file_to(url, path):
    r = requests.get(url)
    with open(path, 'wb') as fout:
        fout.write(r.content)


def load_predictor():
    if not os.path.exists('bayes.pickle'):
        url = os.environ['BAYES_SHARED']
        print("Getting bayes file from %s" % url)
        download_file_to(url, '.\\bayes.pickle')    
    model = from_pickle('bayes.pickle')
    if not os.path.exists('vectorizer.pickle'):
        url = os.environ['VECTORIZER_SHARED']
        print("Getting vectorizer file from %s" % url)
        download_file_to(url, '.\\vectorizer.pickle')    
    vectorizer = from_pickle('vectorizer.pickle')
    if not os.path.exists('label_encoder.pickle'):
        url = os.environ['LABEL_ENCODER_SHARED']
        print("Getting label_encoder file from %s" % url)
        download_file_to(url, '.\\label_encoder.pickle')    
    label_encoder = from_pickle('label_encoder.pickle')
    predictor = Predictor(model, vectorizer, label_encoder)
    return predictor

predictor = load_predictor()

@app.route("/", methods=['GET','POST'])
def hello():
    if request.method == 'GET':
        message = "This is front page"
    else:
        content = request.form['content']
        prediction = predictor.predict(content)
        message = str(prediction)
    return render_template("index.html", message = message)


@app.before_request
def pre_request_logging():
    #Logging statement
    # if 'text/html' in request.headers['Accept']:
    app.logger.info('\t'.join([
        datetime.datetime.today().ctime(),
        request.remote_addr,
        request.method,
        request.url,
        # str(request.data,
        # ', '.join([': '.join(x) for x in request.headers])]
        ])
    )

if __name__ == "__main__":
    app.run()