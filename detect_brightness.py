import numpy as np
import cv2
def detect_brightness(image,x,y,w,h):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x_center,y_center,w,h = map(int, [x, y, w, h])
    x1 = max(0, x_center - w // 2)
    y1 = max(0, y_center - h // 2)
    x2 = min(image.shape[1], x_center + w // 2)
    y2 = min(image.shape[0], y_center + h // 2)
    roi = gray[y1:y2, x1:x2]
    return np.mean(roi)