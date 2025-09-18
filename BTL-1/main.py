from argparse import ArgumentParser
import cv2
import numpy as np
from utils.sinus import auto_remove_sinus
from utils.enhance_contrast import gamma_correction
from utils.watershed import watershed_selective_contours
from utils.logging import logging_step, logging_contours
from utils.binary import *

from settings import settings



def preprocess(img, file_name: str = ""):
    # convert image to gray scale:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    logging_step(img, "grayscale", file_name)

    # remove sinus noise
    img = auto_remove_sinus(img, file_name=file_name, r=3)
    logging_step(img, "remove_sinus", file_name)

    # remove salt and ppepper noise
    img = cv2.medianBlur(img, 3)
    logging_step(img, "remove_salt_pepper", file_name)

    # enhance contrast
    img, gamma = gamma_correction(img)
    logging_step(img, f"enhance_contrast_{gamma}", file_name)

    return img


def obj_counter(img, root_img=None, file_name: str = ""):
    # 1. Convert to binary img
    binary_img = binarize_otsu(img)
    logging_step(binary_img, "binary_img", file_name)

    # 2. xử lý hình thái học
    kernel = np.ones((3, 3), np.uint8)
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=2)
    logging_step(binary_img, "morphology", file_name)

    # 3 get contours of objects
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. Separate contours stuck together
    contours = watershed_selective_contours(binary_img, contours, root_img=root_img.copy())

    # 5. remove min object
    min_area = settings.MIN_AREA  # tùy chỉnh theo ảnh
    obj_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    # log result
    if file_name:
        logging_contours(root_img, obj_contours, file_name=file_name)

    n_objects = len(obj_contours)
    print("Số lượng hạt gạo:", n_objects)

    return n_objects


def run(file_path: str) -> int:
    """Main run count rice in image"""
    print(f"===============STARTING: {file_path}========================")
    settings.step = 0
    file_name = file_path.split("/")[-1].replace(".png", "")
    root_img = cv2.imread(file_path)

    img = root_img.copy()
    logging_step(img, "root_image", file_name)

    # Process flow
    processed_img = preprocess(img, file_name=file_name)

    # object counter
    n_objs = obj_counter(processed_img, root_img, file_name=file_name)

    return n_objs



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", default="data/1_wIXlvBeAFtNVgJd49VObgQ.png")
    args = parser.parse_args()

    # run(args.file)

    run("data/1_wIXlvBeAFtNVgJd49VObgQ.png")
    run("data/1_wIXlvBeAFtNVgJd49VObgQ.png_Salt_Pepper_Noise1.png")
    run("data/1_wIXlvBeAFtNVgJd49VObgQ_sinus.png")
    run("data/1_zd6ypc20QAIFMzrbCmJRMg.png")