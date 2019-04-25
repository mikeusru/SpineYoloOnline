import os

from PIL import Image

from model_data.utils import letterbox_image

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from keras.models import model_from_json
import keras.backend as K
from keras.optimizers import Adam
import numpy as np
import pickle as pkl
from model_data.model import yolo_eval
import time

def get_anchors(anchors_path):
    anchors_path = os.path.expanduser(anchors_path)
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def load_image():
    return None

def resize_image(image):
    return None

def split_image(image):
    return []

def detect_spines_from_image_list(sub_images):
    return boxes, scores

def remove_overlapping_boxes(boxes, scores, image):
    return boxes, scores, image

json_file = open(os.path.join('model_data', 'yolov3_spines.json'), 'r')
loaded_model_from_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_from_json)
loaded_model.load_weights(os.path.join('model_data', 'yolov3_spines_combined_10um_scale.h5'))

input_image_shape = K.placeholder(shape=(2,))
anchors = get_anchors(os.path.join('model_data', 'yolo_anchors.txt'))

start_time = time.time()
boxes, scores, classes = yolo_eval(loaded_model.output, anchors, 1,
                                   input_image_shape, score_threshold=.3,
                                   iou_threshold=.45)
print(" %s seconds " % (time.time()-start_time))

image = load_image()
image_resized = resize_image(image)
sub_images = split_image(image_resized)
boxes = detect_spines_from_image_list(sub_images)
boxes_final, scores_final, image_final = remove_overlapping_boxes(boxes, scores)

# print(boxes)
# image_data = np.array(letterbox_image(Image.open('temp/test_spines.jpg'),(416,416,3)))
# image_data = np.zeros([416, 416, 3])
# image_data = np.expand_dims(image_data, 0)
# yolo_outputs = loaded_model.predict(image_data)
# print(yolo_outputs)
#

with K.get_session() as sess:
    boxes, scores, classes = sess.run([boxes, scores, classes],
                                      feed_dict={
                                          loaded_model.input: image_data,
                                          input_image_shape: [416, 416]})
print(boxes)
