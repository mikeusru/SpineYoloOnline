import os
import time
import ast
import jsonpickle
from flask import Flask, render_template, request, Response
from pusher import Pusher
import jsonpickle.ext.numpy as jsonpickle_numpy

from SpineDetector import SpineDetector

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC_ROOT = os.path.join(APP_ROOT, 'static')
spine_detector = SpineDetector()
spine_detector.set_root_dir(STATIC_ROOT)
spine_detector.start()

jsonpickle_numpy.register_handlers()


def load_pusher_info():
    with open(os.path.join('_private', 'pusher.txt'), 'r') as pusher_inf:
        s = pusher_inf.read()
        pusher_dict = ast.literal_eval(s)
    return pusher_dict


# configure pusher object
pusher_dict = load_pusher_info()
pusher = Pusher(
    app_id=pusher_dict['app_id'],
    key=pusher_dict['key'],
    secret=pusher_dict['secret'],
    cluster=pusher_dict['cluster'],
    ssl=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    spine_detector.set_pusher(pusher)
    uploaded_image_path, u_id = upload_image(request.files.getlist('file'))
    scale = int(request.form['scale'])
    spine_detector.set_inputs(uploaded_image_path, scale)
    spine_detector.queue.put(['find_spines', u_id])
    return render_template("dashboard.html", uID=u_id)


@app.route("/predict-local/<path:img_path>/<float:scale>", methods=['POST'])
def predict_local(img_path, scale):
    u_id = time.strftime("%Y%m%d%H%M%S")
    spine_detector.set_local(True)
    spine_detector.set_inputs(img_path, scale)
    spine_detector.queue.put(['find_spines', u_id, 'local'])
    while u_id not in spine_detector.analyzed_spines.keys():
        continue
    results = spine_detector.analyzed_spines[u_id].tolist()
    response_pickled = jsonpickle.encode(results)
    return Response(response=response_pickled, status=200, mimetype="application/json")


def upload_image(file_list):
    sub_path = 'image_uploads'
    upload_target = os.path.join(STATIC_ROOT, sub_path)
    if not os.path.isdir(upload_target):
        os.makedirs(upload_target)
    for file in file_list:
        filename = file.filename
        timestr = time.strftime("%Y%m%d%H%M%S")
        destination = os.path.join(upload_target, 'image_' + timestr + filename)
        file.save(destination)
    return destination, timestr


@app.route("/submit_training_data", methods=['POST'])
def submit_training_data():
    return render_template("add_training.html")


if __name__ == '__main__':
    print("Current Working Directory ", os.getcwd())
    app.run(host='0.0.0.0', port=5000, debug=False)
