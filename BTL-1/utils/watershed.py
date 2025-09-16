import cv2
import numpy as np
from .logging import logging_step, logging_contours
import cv2
import numpy as np
from settings import settings
import cv2
import numpy as np
from scipy import ndimage as ndi  # cái này vẫn cần cho label


def split_contour(binary_img, contour, contour_index, min_distance=5, root_img=None, dist_thresh=0.6):
    new_contours = []

    # Mask chỉ chứa contour này
    mask = np.zeros(binary_img.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    # Distance transform
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # Tạo marker bằng threshold
    _, sure_fg = cv2.threshold(dist, dist_thresh * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    # logging_step(sure_fg, message=f"sure_fg c_{contour_index}")

    n_markers, markers = cv2.connectedComponents(sure_fg)
    cnts, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if 1000 > cv2.contourArea(c) > settings.MIN_AREA]

    # logging_contours(root_img, cnts, f"watershed_contours_{contour_index}")
    new_contours.extend(cnts)

    return new_contours



def watershed_selective_contours(binary_img, contours, factor=1.5, root_img=None):
    new_contours = []

    # Tính diện tích trung bình
    areas = [cv2.contourArea(c) for c in contours]
    mean_area = np.mean(areas)

    for i, c in enumerate(contours):
        area = cv2.contourArea(c)

        # Mask cho contour
        mask = np.zeros_like(binary_img)
        cv2.drawContours(mask, [c], -1, 255, -1)
        roi_mask = cv2.bitwise_and(binary_img, mask)

        if area < factor * mean_area:
            # Contour nhỏ: giữ nguyên
            new_contours.append(c)
            continue

        # print(i, area, mean_area, area/mean_area)
        new_contours.extend(split_contour(binary_img, c, i, min_distance=5, root_img=root_img))

    return new_contours
