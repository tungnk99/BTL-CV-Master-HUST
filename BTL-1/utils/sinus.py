import numpy as np


import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.logging import logging_step

def detect_sinusoidal_noise(img, threshold=10):
    # FFT
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))

    # Normalize để dễ threshold
    norm_mag = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Tìm các điểm sáng vượt ngưỡng
    _, mask = cv2.threshold(norm_mag, threshold, 255, cv2.THRESH_BINARY)

    # Loại bỏ trung tâm (DC component)
    h, w = mask.shape
    cy, cx = h//2, w//2
    cv2.circle(mask, (cx, cy), 20, 0, -1)  # xóa vùng trung tâm 20px

    # Đếm số điểm sáng còn lại
    n_points = cv2.countNonZero(mask)

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1); plt.imshow(img, cmap='gray'); plt.title("Ảnh gốc"); plt.axis("off")
    plt.subplot(1,2,2); plt.imshow(norm_mag, cmap='gray'); plt.title("Magnitude Spectrum"); plt.axis("off")
    plt.show()

    print(f"Detected bright spots (excluding center): {n_points}")

    return n_points > 0   # True nếu có nhiễu sinus


def remove_sinus(img):
    # FFT
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(1 + np.abs(fshift))

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)

    r = 15
    mask[crow - 50 - r:crow - 50 + r, ccol - 0 - r:ccol - 0 + r] = 0
    mask[crow + 50 - r:crow + 50 + r, ccol - 0 - r:ccol - 0 + r] = 0

    fshift_filtered = fshift * mask

    # IFFT
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = img_back.astype(np.uint8)

    return img_back


def auto_remove_sinus(gray, r=5, thresh_ratio=2.5, file_name: str = ""):
    """
    Tự động loại bỏ nhiễu dạng sinusoidal bằng FFT.

    gray         : ảnh grayscale (numpy array)
    r            : bán kính vùng che tại mỗi đỉnh phổ
    thresh_ratio : hệ số so với trung bình phổ để chọn điểm sáng
    """
    logging_step(gray, message="before sinus", file_name=file_name)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    # FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    spectrum = np.log(1 + np.abs(fshift)).astype(np.float32)

    # Normalize để hiển thị & threshold
    spectrum_disp = cv2.normalize(spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # logging_step(spectrum_disp, message="spectrum", file_name="sinus_test")

    # Threshold trên phổ normalize
    mean_val = np.mean(spectrum_disp)
    max_val = np.max(spectrum_disp)
    thresh = mean_val * thresh_ratio
    _, binary = cv2.threshold(spectrum_disp, thresh, 255, cv2.THRESH_BINARY)
    logging_step(binary, message="spectrum_disp_binary", file_name=file_name)

    # Contours = các đốm sáng
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # logging_contours(spectrum_disp, contours, file_name="sinus_test")

    if len(contours) < 3:
        return gray

    # Mask
    mask = np.ones((rows, cols), np.uint8)

    for cnt in contours:
        (x, y), _ = cv2.minEnclosingCircle(cnt)
        x, y = int(x), int(y)

        # bỏ qua tâm (tần số thấp)
        if x == ccol and y == crow:
            continue

        print(x, y)

        # che điểm và điểm đối xứng
        cv2.circle(mask, (x, y), r, 0, -1)
        cv2.circle(mask, (cols - x, rows - y), r, 0, -1)

    # Apply mask
    fshift_filtered = fshift * mask
    img_back = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
    img_back = np.abs(img_back)

    # Chuẩn hóa về 0–255
    return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
