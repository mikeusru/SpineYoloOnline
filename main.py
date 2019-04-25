import os

from PIL import Image

from model_data.utils import letterbox_image

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import keras.backend as K
import time
from SpineDetector import SpineDetector

detector = SpineDetector()
detector.find_spines('temp/test_spines.jpg', 10)