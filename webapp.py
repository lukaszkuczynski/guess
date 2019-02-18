from flask import Flask, render_template, request, redirect
import datetime, logging
from logging import handlers
from predict import Predictor, from_pickle
import os, requests
import tempfile


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

def load_predictor():
    current_dir = os.path.curdir
    local_path = os.path.join(current_dir, 'bayes.pickle')
    if not os.path.exists(local_path):
        raise Exception("model file missing at '%s'" % local_path)
    model = from_pickle(local_path)

    local_path = os.path.join(current_dir, 'vectorizer.pickle')
    if not os.path.exists(local_path):
        raise Exception("vectorizer file missing at '%s'" % local_path)
    vectorizer = from_pickle(local_path)

    local_path = os.path.join(current_dir, 'label_encoder.pickle')
    if not os.path.exists(local_path):
        raise Exception("labelencoder file missing at '%s'" % local_path)
    label_encoder = from_pickle(local_path)

    predictor = Predictor(model, vectorizer, label_encoder)
    return predictor


predictor = load_predictor()

@app.route("/", methods=['GET','POST'])
def hello():
    categories = None
    values = None
    predicted_category = None
    error_message = None
    if request.method == 'GET':
        pass
    else:
        content = request.form['content']
        if not content.strip():
            error_message = "Cannot predict empty content!"
        else:
            prediction = predictor.predict(content)
            sorted_categories = sorted(prediction.items(), key=lambda tup:tup[1], reverse=True)
            categories = list(dict(sorted_categories).keys())
            values = list(dict(sorted_categories).values())
            predicted_category = categories[0]
    return render_template("index.html", categories=categories, values=values, predicted_category=predicted_category, error_message=error_message)


@app.before_request
def pre_request_logging():
    app.logger.info('\t'.join([
        datetime.datetime.today().ctime(),
        request.remote_addr,
        request.method,
        request.url,
        ])
    )

if __name__ == "__main__":
    app.run()