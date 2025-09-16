import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Ví dụ chạy thử
if __name__ == "__main__":
    img = cv2.imread("data/1_wIXlvBeAFtNVgJd49VObgQ.png_Salt_Pepper_Noise1.png", cv2.IMREAD_GRAYSCALE)
    plot_histogram(img, "Histogram ảnh")
