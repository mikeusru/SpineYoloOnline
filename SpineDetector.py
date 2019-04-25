import os

import numpy as np
from keras.models import model_from_json
import keras.backend as K
from skimage import transform

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
        self.target_scale_pixels_per_um = 10
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

    def _rescale_image(self, image, original_scale):
        resize_ratio = self.target_scale_pixels_per_um / original_scale
        new_shape = np.array(image.shape)
        new_shape[:2] = np.array(new_shape[:2] * resize_ratio, dtype=np.int)
        image_rescaled = transform.resize(image, new_shape, preserve_range=True).astype(np.uint8)
        return image_rescaled, resize_ratio



    def _preprocess_window(self, image):
        boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        image_data = np.array(boxed_image, dtype='float32')
        print('Shape: {}, max: {}'.format(image_data.shape, image_data.max()))
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        return image_data

    def _get_sliding_window_indices(self, image_shape):
        window_size = self.model_image_size
        step_size = int(self.model_image_size[0] / 2)
        window_list = []
        for row_ind in range(0, image_shape[0], step_size):
            for col_ind in range(0, image_shape[1], step_size):
                rc = {'r': row_ind, 'c': col_ind,
                      'r_max': min(row_ind + window_size[0], image_shape[0]),
                      'c_max': min(col_ind + window_size[1], image_shape[1])}
                window_list.append(rc)
        return window_list

    def _detect_spines_in_windows(self, image, window_list):
        for window in window_list:
            image_cut = image[window['r']:window['r_max'], window['c']:window['c_max']]
            image_data = self._preprocess_window(image_cut)
            # TODO: Should I run this on batch images because there's a batch dimension?
            with K.get_session() as sess:
                boxes_out, scores_out, classes_out = sess.run([self.boxes, self.scores, self.classes],
                                                              feed_dict={
                                                                  self.model.input: image_data,
                                                                  self.input_image_shape: [416, 416]})
                window['boxes'] = np.array(boxes_out)
                window['scores'] = np.array(scores_out)
        return window_list

    def _shift_boxes(self, window_list):
        for window in window_list:
            window['boxes'][:, [0, 2]] += window['r']
            window['boxes'][:, [1, 3]] += window['c']
        return window_list

    def run(self, image, scale):
        image_rescaled, resize_ratio = self._rescale_image(image, scale)
        window_list = self._get_sliding_window_indices(image_rescaled.shape[:2])
        window_list = self._detect_spines_in_windows(image_rescaled, window_list)
        window_list = self._shift_boxes(window_list)
        window_list = self._rescale_boxes(window_list)
        boxes, scores = self._remove_overlapping_boxes(window_list)
        r_image = self._draw_output_image(image, boxes, scores)
        return r_image, boxes, scores