import cv2
import numpy as np
from utils.logging import logging_step, logging_contours


def auto_remove_sinus(gray, r=10, thresh_ratio=2.5):
    """
    Tự động loại bỏ nhiễu dạng sinusoidal bằng FFT.

    gray         : ảnh grayscale (numpy array)
    r            : bán kính vùng che tại mỗi đỉnh phổ
    thresh_ratio : hệ số so với trung bình phổ để chọn điểm sáng
    """
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    # FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    spectrum = np.log(1 + np.abs(fshift)).astype(np.float32)

    # Normalize để hiển thị & threshold
    spectrum_disp = cv2.normalize(spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    logging_step(spectrum_disp, message="spectrum", file_name="sinus_test")

    # Threshold trên phổ normalize
    mean_val = np.mean(spectrum_disp)
    max_val = np.max(spectrum_disp)
    thresh = mean_val * thresh_ratio
    _, binary = cv2.threshold(spectrum_disp, thresh, 255, cv2.THRESH_BINARY)
    logging_step(binary, message="binary", file_name="sinus_test")

    # Contours = các đốm sáng
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logging_contours(spectrum_disp, contours, file_name="sinus_test")

    # Mask
    mask = np.ones((rows, cols), np.uint8)


    for cnt in contours:
        (x, y), _ = cv2.minEnclosingCircle(cnt)
        x, y = int(x), int(y)

        # bỏ qua tâm (tần số thấp)
        if x == ccol and y == crow:
            continue

        # che điểm và điểm đối xứng
        cv2.circle(mask, (x, y), r, 0, -1)
        cv2.circle(mask, (cols - x, rows - y), r, 0, -1)

    # Apply mask
    fshift_filtered = fshift * mask
    img_back = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
    img_back = np.abs(img_back)

    # Chuẩn hóa về 0–255
    return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


if __name__ == '__main__':
    img = cv2.imread("data/1_wIXlvBeAFtNVgJd49VObgQ_sinus.png", 0)
    denoised = auto_remove_sinus(img, r=5, thresh_ratio=2.5)

    cv2.imwrite("denoised_auto.png", denoised)
    cv2.imshow("Result", denoised)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
