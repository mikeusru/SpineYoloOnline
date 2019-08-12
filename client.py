# testing local client for data xfer
from __future__ import print_function
import requests
import json

addr = 'http://localhost:5000'

img_path = 'C:\\Users\\smirnovm\\Documents\\Data\\yolo_spine_training\\images\\000001.jpg'
scale = 9.0

test_url = addr + '/predict-local/{}/{}'.format(img_path, scale)

response = requests.post(test_url)
# decode response
# print(response)
print(json.loads(response.text))
