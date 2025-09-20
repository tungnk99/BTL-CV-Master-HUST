import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Hàm remove sinusoidal noise với mask tự chọn ---
def remove_sinus(gray, points, r=25):
    """
    gray   : ảnh grayscale
    points : danh sách tọa độ điểm nhiễu (ccol + d, crow) hoặc (x, y)
    r      : bán kính mask (cố định cho tất cả các điểm)
    """
    # FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    rows, cols = gray.shape
    mask = np.ones((rows, cols), np.uint8)

    # Vẽ mask ở các điểm chọn với bán kính r
    for (x, y) in points:
        cv2.circle(mask, (x, y), r, 0, -1)

    # Áp dụng mask
    fshift_filtered = fshift * mask

    # Biến đổi ngược
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # Chuẩn hóa về 0–255
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img_back


# --- Tool chọn điểm bằng chuột ---
points = []

def onclick(event):
    if event.inaxes:
        x, y = int(event.xdata), int(event.ydata)
        points.append((x, y))
        print(f"Chọn điểm: {(x, y)}")

if __name__ == "__main__":
    # Đọc ảnh
    img = cv2.imread("data/1_wIXlvBeAFtNVgJd49VObgQ_sinus.png", 0)

    # FFT spectrum
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    spectrum = np.log(1 + np.abs(fshift))

    # Hiển thị phổ và click chọn
    fig, ax = plt.subplots()
    ax.imshow(spectrum, cmap='gray')
    ax.set_title("Click vào các điểm nhiễu trong phổ FFT")
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    # Sau khi đóng cửa sổ sẽ xử lý
    if points:
        denoised = remove_sinus(img, points, r=5)  # tăng r để che rộng hơn
        cv2.imwrite("denoised.png", denoised)
        cv2.imshow("Result", denoised)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
