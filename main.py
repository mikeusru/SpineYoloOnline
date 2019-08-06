import os
import time
import numpy as np
from flask import Flask, render_template, request
from flask_jsglue import JSGlue
from pusher import Pusher

from SpineDetector import SpineDetector
from ThreadTest import MyThread

app = Flask(__name__)
jsglue = JSGlue(app)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC_ROOT = os.path.join(APP_ROOT, 'static')
spine_detector = SpineDetector()
spine_detector.set_root_dir(STATIC_ROOT)
spine_detector.start()

# configure pusher object
pusher = Pusher(
    app_id='837983',
    key='309568955ad8ba7e672c',
    secret='13e738647a96b49408a6',
    cluster='us2',
    ssl=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    spine_detector.set_pusher(pusher)
    uploaded_image_path = upload_image(request.files.getlist('file'))
    scale = int(request.form['scale'])
    spine_detector.set_inputs(uploaded_image_path, scale)
    spine_detector.queue.put("find_spines")
    # r_image, r_boxes = spine_detector.find_spines(uploaded_image_path, scale)
    # image_file, data_file = save_results(r_image, r_boxes)
    # print('detection done')
    # return render_template("results.html", boxes=r_boxes, image_name=image_file, data_name=data_file)
    return render_template("dashboard.html")


# @app.route("/predict", methods=['POST'])
# def predict():
#     my_thread_obj1 = MyThread(4, pusher)
#     my_thread_obj1.start()
#     return render_template("dashboard.html")


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
    return destination


@app.route("/submit_training_data", methods=['POST'])
def submit_training_data():
    return render_template("add_training.html")


if __name__ == '__main__':
    print("Current Working Directory ", os.getcwd())
    app.run(host='0.0.0.0', port=5000, debug=False)
