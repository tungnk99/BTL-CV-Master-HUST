import cv2
import numpy as np


def laplacian_contrast(img):
    lap = cv2.Laplacian(img, cv2.CV_64F)
    return lap.var()


def compute_contrast(img_path):
    print("-------------------")
    print(img_path)
    img = cv2.imread(img_path)
    mean_val = img.mean()
    std_val = img.std()

    print("Std contrast:", std_val)
    print("Mean contrast:", mean_val)


if __name__ == '__main__':
    # compute_contrast("data/1_wIXlvBeAFtNVgJd49VObgQ.png")
    # compute_contrast("data/1_wIXlvBeAFtNVgJd49VObgQ.png_Salt_Pepper_Noise1.png")
    # compute_contrast("data/1_wIXlvBeAFtNVgJd49VObgQ_sinus.png")
    # compute_contrast("data/1_zd6ypc20QAIFMzrbCmJRMg.png")
    compute_contrast("logs/1_wIXlvBeAFtNVgJd49VObgQ_sinus/6.remove salt and pepper.png")