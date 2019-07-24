from SpineDetector import SpineDetector

IMG_PATH = 'test/tiff_stack.tif'
SCALE = 15

sp = SpineDetector()
r_image, r_boxes = sp.find_spines(IMG_PATH, SCALE)
