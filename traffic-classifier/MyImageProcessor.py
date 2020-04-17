import cv2
import numpy as np

from skimage import data
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.transform import warp, AffineTransform, ProjectiveTransform

class MyImageProcessor(object):

    # noinspection PyMethodMayBeStatic
    def apply_grayscale_and_normalize(self, image):
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html
        clahe_histogram = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = clahe_histogram.apply(gray_image)

        # Apply min max scaling
        gray_image = (np.float32(gray_image) - np.min(gray_image)) / (np.max(gray_image) -  np.min(gray_image))
        return gray_image

    # noinspection PyMethodMayBeStatic
    def apply_warp_augmentation_on_an_image(self, image):
        image_size = image.shape[0]
        #aff_tform = AffineTransform(scale=(1, 1/1.2), rotation=1, shear=0.7, translation=(210, 50))
        aff_tform = AffineTransform(scale=(1, 1 / 1.2))
        image = warp(image, aff_tform, output_shape=(image_size, image_size), order = 1, mode = 'edge')

        x = image_size * 0.2
        # top_right - x,y
        # bottom_left - x,y
        # top_right - x,y
        # bottom_right - x,y
        tl_x, tl_y, bl_x, bl_y, tr_x, tr_y, br_x, br_y = np.random.uniform(-x, x, size=8)

        src = np.array([[tl_y, tl_x], [bl_y, image_size-bl_x], [image_size-br_y, image_size-br_x], [image_size-tr_y, tr_x]])
        dst = np.array([[0, 0], [0, image_size], [image_size, image_size], [image_size, 0]])

        proj_tform = ProjectiveTransform()
        proj_tform.estimate(src, dst)

        image = warp(image, proj_tform, output_shape=(image_size, image_size), order = 1, mode = 'edge')

        return image



