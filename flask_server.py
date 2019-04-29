import os
import time
import numpy as np
from flask import Flask, render_template, request

from SpineDetector import SpineDetector

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC_ROOT = os.path.join(APP_ROOT, 'static')


@app.route('/')
def index():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    spine_detector = SpineDetector()
    uploaded_image_path = upload_image(request.files.getlist('file'))
    scale = int(request.form['scale'])
    r_image, r_boxes = spine_detector.find_spines(uploaded_image_path, scale)
    image_file, data_file = save_results(r_image, r_boxes)
    print('detection done')
    return render_template("results.html", boxes=r_boxes, image_name=image_file, data_name=data_file)


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


def save_results(image, boxes):
    sub_path = 'results'
    if not os.path.isdir(os.path.join(STATIC_ROOT, sub_path)):
        os.makedirs(os.path.join(STATIC_ROOT, sub_path))
    timestr = time.strftime("%Y%m%d%H%M%S")
    img_path_relative = os.path.join(sub_path, 'r_img' + timestr + '.jpg')
    image_path_full = os.path.join(STATIC_ROOT, img_path_relative)
    image.save(image_path_full)
    boxes_path_relative = os.path.join(sub_path, 'r_boxes' + timestr + '.csv')
    boxes_path_full = os.path.join(STATIC_ROOT, boxes_path_relative)
    np.savetxt(boxes_path_full, boxes, delimiter=',')
    return img_path_relative, boxes_path_relative


@app.route("/submit_training_data", methods=['POST'])
def submit_training_data():
    return render_template("add_training.html")


if __name__ == '__main__':
    print("Current Working Directory ", os.getcwd())
    app.run(host='0.0.0.0', port=5000, debug=False)
