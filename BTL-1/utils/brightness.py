import cv2
import numpy as np

def check_reduce_brightness(img, mean_thresh=100, std_thresh=50):
    """
    Kiểm tra xem ảnh có cần giảm sáng không dựa vào mean và std.
    Trả về True nếu cần giảm sáng, False nếu không.
    """
    mean_val = np.mean(img)
    std_val = np.std(img)
    print(f"Mean: {mean_val:.2f}, Std: {std_val:.2f}")

    # Quyết định giảm sáng: mean quá cao hoặc std quá cao
    if mean_val > mean_thresh:
        return True
    return False

def reduce_brightness(img, factor=0.7):
    """
    Giảm sáng ảnh bằng cách nhân với factor (0-1)
    """
    img_adjusted = np.clip(img * factor, 0, 255).astype(np.uint8)
    mean_val = np.mean(img_adjusted)
    std_val = np.std(img_adjusted)
    print(f"After deduce bright. Mean: {mean_val:.2f}, Std: {std_val:.2f}")
    return img_adjusted