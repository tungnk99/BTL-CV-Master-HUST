from pydantic.dataclasses import dataclass
from argparse import ArgumentParser
import cv2
import os
import numpy as np
from utils.sinus import remove_sinus, auto_remove_sinus
from utils.enhance_contrast import gamma_correction
from utils.watershed import watershed_selective_contours
from utils.logging import logging_step, logging_contours
from utils.brightness import check_reduce_brightness, reduce_brightness
from settings import settings



def preprocess(img, file_name: str = ""):
    # convert image to gray scale:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    logging_step(img, "grayscale", file_name)


    # # remove sinus noise
    img = auto_remove_sinus(img, file_name=file_name)
    logging_step(img, "remove sinus", file_name)


    img = cv2.medianBlur(img, 3)
    logging_step(img, "remove salt and pepper", file_name)

    # enhance_contrast
    img = gamma_correction(img, gamma=0.2)
    logging_step(img, "enhance contrast", file_name)

    # remove salt & pepper noise
    return img


def obj_counter(img, root_img=None, file_name: str = ""):
    # 1. Convert to binary img
    _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    logging_step(binary_img, "binary img", file_name)

    # Local Adaptive Thresholding
    binary_img = cv2.adaptiveThreshold(binary_img, 255.0, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, -20.0)
    logging_step(binary_img, "local adaptive", file_name)


    # 2. xử lý hình thái học
    kernel = np.ones((3, 3), np.uint8)
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=2)
    logging_step(binary_img, "morpology", file_name)

    # 3 get contours of objects
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = watershed_selective_contours(binary_img, contours, root_img=root_img.copy())

    # 3. remove min object
    min_area = settings.MIN_AREA  # tùy chỉnh theo ảnh
    obj_contours = [c for c in contours if cv2.contourArea(c) > min_area]


    if file_name:
        logging_contours(root_img, obj_contours, file_name=file_name)

    n_objects = len(obj_contours)
    print("Số lượng hạt gạo:", n_objects)

    return n_objects


def run(file_path: str) -> int:
    """Main run count rice in image"""
    print("=======================================", file_path)
    settings.step = 0
    file_name = file_path.split("/")[-1].replace(".png", "")
    root_img = cv2.imread(file_path)

    img = root_img.copy()
    logging_step(img, "root image", file_name)

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