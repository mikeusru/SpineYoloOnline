import os
import time
from SpineDetector import SpineDetector

SPINE_DETECTOR = None
str_bucket = 'spine-yolo-serverless'


def handler(event, context):
    global str_bucket
    if not os.path.exists('/temp/images'):
        os.makedirs('/temp/images')
    if not os.path.exists('/temp/model'):
        os.makedirs('/temp/model')

    str_file = '/temp/images/test_spines.jpg'

    download_from_s3(str_bucket, '/images/test_spines.jpg', str_file)

    global SPINE_DETECTOR
    if SPINE_DETECTOR is None:
        download_from_s3(str_bucket, 'model/yolov3_spines.json', '/temp/model/model.json')
        download_from_s3(str_bucket, 'model/yolov3_spines_combined_10um_scale.h5', '/temp/model/weights.h5')
    image = os.path.join('/temp/images/', 'test_spines.jpg')
    result = run_inference_on_image(image)
    return result


def run_inference_on_image(image):
    global SPINE_DETECTOR
    if SPINE_DETECTOR is None:
        SPINE_DETECTOR = SpineDetector()
    r_image, boxes_scores = SPINE_DETECTOR.find_spines('temp/images/test_spines.jpg', 10)
    return boxes_scores


def download_from_s3(strBucket, strKey, strFile):
    s3_client = boto3.client('s3')
    s3_client.download_file(strBucket, strKey, strFile)


def save_results(image, boxes):
    sub_path = 'results'
    if not os.path.isdir(sub_path):
        os.makedirs(sub_path)
    timestr = time.strftime("%Y%m%d%H%M%S")
    img_path_relative = os.path.join(sub_path, 'r_img' + timestr + '.jpg')
    image.save(img_path_relative)

# detector = SpineDetector()
# r_image, boxes_scores = detector.find_spines('temp/000002.tif', 10)
# save_results(r_image, boxes_scores)
