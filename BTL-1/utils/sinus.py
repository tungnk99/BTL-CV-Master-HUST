import numpy as np


import cv2
import numpy as np
import matplotlib.pyplot as plt

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