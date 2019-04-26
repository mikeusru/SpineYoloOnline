import os
import time
from SpineDetector import SpineDetector


strBucket = 'spine-yolo-serverless'

def save_results(image, boxes):
    sub_path = 'results'
    if not os.path.isdir(sub_path):
        os.makedirs(sub_path)
    timestr = time.strftime("%Y%m%d%H%M%S")
    img_path_relative = os.path.join(sub_path, 'r_img' + timestr + '.jpg')
    image.save(img_path_relative)


detector = SpineDetector()
r_image, boxes_scores = detector.find_spines('temp/000002.tif', 10)
# save_results(r_image, boxes_scores)
