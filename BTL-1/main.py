from pydantic.dataclasses import dataclass
from argparse import ArgumentParser
import cv2
import os
import numpy as np
from utils.sinus_utils import remove_sinus
from utils.enhance_contrast import enhance_contrast


@dataclass
class Setting:
    DEBUG_SHOW: int = 0
    DEBUG_SAVE_IMG: int = 1
    LOG_DIRS: str = "logs"


settings = Setting()


def logging(img, message: str = "", file_name: str = ""):
    if settings.DEBUG_SHOW:
        cv2.imshow(message, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if settings.DEBUG_SAVE_IMG:
        if file_name:
            log_dir = f"{settings.LOG_DIRS}/{file_name}"
        else:
            log_dir = settings.LOG_DIRS

        os.makedirs(log_dir, exist_ok=True)
        cv2.imwrite(f"{log_dir}/{message}.png", img)


def preprocess(img, file_name: str = ""):
    # convert image to gray scale:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    logging(img, "2. grayscale", file_name)

    # remove salt & pepper noise
    img = cv2.medianBlur(img, 5)
    logging(img, "3. after rm salt and pepper noise", file_name)

    # remove sinus noise
    img = remove_sinus(img)
    logging(img, "3. after rm sinus", file_name)

    # enhance_contrast
    img = enhance_contrast(img, method="gamma")
    logging(img, "4. enhance contrast", file_name)
    return img


def obj_counter(img, file_name: str = ""):
    # 1. Convert to binary img
    _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. xử lý hình thái học
    kernel = np.ones((3, 3), np.uint8)
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=2)

    logging(binary_img, "5. binary img", file_name)

    # 3 get contours of objects
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 3. remove min object
    min_area = 50  # tùy chỉnh theo ảnh
    obj_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    img_results = img.copy()
    cv2.drawContours(img_results, obj_contours, -1, (0, 255, 0), 2)
    logging(img_results, "6. image results", file_name)

    n_objects = len(obj_contours)
    print("Số lượng hạt gạo:", n_objects)

    return n_objects

def run(file_path: str) -> int:
    """Main run count rice in image"""
    file_name = file_path.split("/")[-1].replace(".png", "")
    img = cv2.imread(file_path)
    logging(img, "1. root image", file_name)

    # Process flow
    img = preprocess(img, file_name=file_name)

    # object counter
    n_objs = obj_counter(img, file_name=file_name)

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