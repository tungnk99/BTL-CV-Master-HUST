import cv2
import numpy as np


def enhance_contrast(img, method="clahe"):
    if method == "hist_eq":
        return cv2.equalizeHist(img)
    elif method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        return clahe.apply(img)
    elif method == "gamma":
        gamma = 1.5
        return np.array(255*(img/255.0)**(1/gamma), dtype='uint8')
    else:
        raise ValueError("Unknown method")
