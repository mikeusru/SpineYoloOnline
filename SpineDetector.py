import os

import numpy as np
from keras.models import model_from_json
import keras.backend as K

from model_data.model import yolo_eval
from model_data.utils import letterbox_image


class SpineDetector:
    def __init__(self):
        self.anchors_path = os.path.join('model_data', 'yolo_anchors.txt')
        self.model_path = os.path.join('model_data', 'yolov3_spines.json')
        self.weights_path = os.path.join('model_data', 'yolov3_spines_combined_10um_scale.h5')
        self.class_names = ['Spine']
        self.score_threshold = 0.3
        self.iou_threshold = 0.45
        self.model_image_size = (416, 416)
        self.anchors = self._load_anchors()
        self.model = self._load_model()
        self.boxes, self.scores, self.classes = self._generate_output_tensors()

    def _load_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def _load_model(self):
        json_file = open(os.path.join(self.model_path), 'r')
        loaded_model_from_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_from_json)
        loaded_model.load_weights(self.weights_path)
        print('loaded model: {}\n loaded weights: {}'.format(self.model_path, self.weights_path))
        return loaded_model

    def _generate_output_tensors(self):
        self.input_image_shape = K.placeholder(shape=(2,))
        boxes, scores, classes = yolo_eval(self.model.output, self.anchors, len(self.class_names),
                                           self.input_image_shape, score_threshold=self.score_threshold,
                                           iou_threshold=self.iou_threshold)
        return boxes, scores, classes

    def detect_spines_in_image_list(self, image_list):
        boxes = []
        scores = []
        for image in image_list:
            image_data = self._preprocess_image(image)
            # TODO: Should I run this on batch images because there's a batch dimension?
            with K.get_session() as sess:
                boxes_out, scores_out, classes_out = sess.run([self.boxes, self.scores, self.classes],
                                                              feed_dict={
                                                                  self.model.input: image_data,
                                                                  self.input_image_shape: [416, 416]})
                boxes.append(boxes_out)
                scores.append(scores_out)
        return boxes, scores

    def _preprocess_image(self, image):
        boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        image_data = np.array(boxed_image, dtype='float32')
        print('Shape: {}, max: {}'.format(image_data.shape, image_data.max()))
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        return image_data