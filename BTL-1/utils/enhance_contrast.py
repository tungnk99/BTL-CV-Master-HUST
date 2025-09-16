import numpy as np


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

    return 0.67


def gamma_correction(img, gamma=None):
    normalized = img / 255.0

    if not gamma:
        gamma = get_gamma(img)

    corrected = np.power(normalized, gamma)
    img = np.uint8(corrected * 255)
    return img

