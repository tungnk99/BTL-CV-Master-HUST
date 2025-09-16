import cv2
import matplotlib.pyplot as plt


def plot_histogram(img, title="Histogram", show: bool = True, save_path: str = ""):
    # Compute histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(img, cmap="gray")
    plt.title("Root image")
    plt.axis("off")

    # Histogram
    plt.subplot(1,2,2)
    plt.plot(hist, color="black")
    plt.title(title)
    plt.xlim([0,256])
    plt.xlabel("Giá trị pixel")
    plt.ylabel("Số lượng pixel")

    plt.tight_layout()

    if show:
        plt.show()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')