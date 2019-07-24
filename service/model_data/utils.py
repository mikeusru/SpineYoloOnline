"""Miscellaneous utility functions."""

from functools import reduce
from PIL import Image
import numpy as np


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image, scale


def pad_image(image, size):
    '''pad image to size'''
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, (0, 0))
    return new_image, 1


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def calc_iou(boxA, boxB):
    # make sure boxB is a two-dimensional array
    if len(boxB.shape) == 1:
        boxB = np.expand_dims(boxB, axis=0)
    # determine the (x, y)-coordinates of the intersection rectangle
    # box is top left bottom right
    # or more like bottom left top right... just works out this way bc image is flipped or something
    xA = np.maximum(boxA[1], boxB[:, 1])
    yA = np.maximum(boxA[0], boxB[:, 0])
    xB = np.minimum(boxA[3], boxB[:, 3])
    yB = np.minimum(boxA[2], boxB[:, 2])

    # compute the area of intersection rectangle
    interArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)

    # compute the area of both rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[:, 2] - boxB[:, 0] + 1) * (boxB[:, 3] - boxB[:, 1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / (boxAArea + boxBArea - interArea)

    return iou


def load_tiff_stack(path):
    count = 0
    img = Image.open(path)
    while True:
        try:
            img.seek(count)
        except EOFError:
            break
        count += 1
    return img, count


def _max_projection_from_list(image_list):
    if len(image_list) != 1:
        image_max = np.max(np.stack(image_list, axis=2), axis=2)
    else:
        image_max = image_list[0]
    return image_max
