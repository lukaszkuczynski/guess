from flask import Flask, render_template, request, redirect
import datetime, logging
from logging import handlers

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

@app.route("/", methods=['GET','POST'])
def hello():
    if request.method == 'GET':
        message = "This is front page"
    else:
        content = request.form['content']
        message = 'WAS POSTED'
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