import os

# from PIL import Image

# from model_data.utils import letterbox_image

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# import keras.backend as K
import time
from SpineDetector import SpineDetector

def save_results(image, boxes):
    sub_path = 'results'
    if not os.path.isdir(sub_path):
        os.makedirs(sub_path)
    timestr = time.strftime("%Y%m%d%H%M%S")
    img_path_relative = os.path.join(sub_path, 'r_img' + timestr + '.jpg')
    image.save(img_path_relative)
    # boxes_path_relative = os.path.join(sub_path, 'r_boxes' + timestr + '.csv')
    # boxes_path_full = os.path.join(STATIC_ROOT, boxes_path_relative)
    # np.savetxt(boxes_path_full, boxes, delimiter=',')
    # return img_path_relative, boxes_path_relative

detector = SpineDetector()
r_image, boxes_scores = detector.find_spines('temp/000002.tif', 10)
save_results(r_image, boxes_scores)
