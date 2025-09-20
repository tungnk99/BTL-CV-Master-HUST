import cv2
import numpy as np
import os
from main import obj_counter


class Setting:
    DEBUG_SHOW: int = 0
    DEBUG_SAVE_IMG: int = 1
    LOG_DIRS: str = "logs_tests2"


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



def hist_e(img):
    img = cv2.equalizeHist(img)
    logging(img, "hit_e")

    return img


def clahe_e(img, clipLimit=2.0):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    logging(img, "clahe_e")

    return img

def get_gamma(img) -> float:
    std: float = img.std()
    mean: float = img.mean()

    if std > 40:
        return 1

    if std < 15 and mean < 15:
        return 0.1

    if std < 20 and mean < 20:
        return 0.2

    if std < 25 and mean < 25:
        return 0.4

    if mean > 200 and std < 30:
        return 4

    if mean > 170 and std < 30:
        return 2

    if mean < 100:
        return 0.8

    elif mean < 150:
        return 1.0

    elif mean < 200:
        return 1.2

    else:
        return 1.5


def gamma_correction(img, gamma=None):
    std: float = img.std()
    mean: float = img.mean()

    normalized = img / 255.0

    if not gamma:
        gamma = get_gamma(img)

    print(mean, std, gamma)
    corrected = np.power(normalized, gamma)
    img = np.uint8(corrected * 255)
    logging(img, f"gamma_correction_{gamma}")

    return img


if __name__ == '__main__':
    img_path = "data/1_zd6ypc20QAIFMzrbCmJRMg.png"
    root_img = cv2.imread(img_path)
    img = cv2.cvtColor(root_img, cv2.COLOR_BGR2GRAY)
    img = gamma_correction(img)

    for gamma in [0.1, 0.2, 0.4, 0.67, 1, 1.5, 2.5, 5, 10, 25]:
        print("----------------------", gamma)
        img = gamma_correction(img, gamma)
        obj_counter(img, root_img=root_img, file_name=f"1_zd6ypc20QAIFMzrbCmJRMg_{gamma}")
