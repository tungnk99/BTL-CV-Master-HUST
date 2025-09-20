import cv2
import os
from settings import settings
from utils.contours import draw_index_contours


def show_img(img, message: str = ""):
    cv2.imshow(message, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def logging_step(img, message: str = "", file_name: str = ""):
    settings.step += 1

    if settings.DEBUG_SHOW:
        show_img()

    if settings.DEBUG_SAVE_IMG:
        if file_name:
            log_dir = f"{settings.LOG_DIRS}/{file_name}"
        else:
            log_dir = settings.LOG_DIRS

        os.makedirs(log_dir, exist_ok=True)

        cv2.imwrite(f"{log_dir}/{settings.step}.{message}.png", img)


def logging_contours(root_img, contours, file_name="contours"):
    img_results = root_img.copy()
    cv2.drawContours(img_results, contours, -1, (0, 255, 0), 2)
    img_results = draw_index_contours(img_results, contours)
    logging_step(img_results, "image_results", file_name)