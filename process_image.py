import os
import cv2
import numpy as np
from glob import glob

def preprocess_image(image):
    # Ensure image has 3 channels before converting
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)

    # Gaussian Blur
    blurred = cv2.GaussianBlur(contrast, (3, 3), 0)

    # Otsu threshold
    _, th2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return th2
