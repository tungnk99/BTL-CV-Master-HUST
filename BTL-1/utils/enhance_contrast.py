import numpy as np


def get_gamma(img) -> float:
    std: float = img.std()
    mean: float = img.mean()

    if std > 50:
        return 1

    if mean < 50:
        return  0.1
    elif mean < 100:
        return 0.2
    elif mean < 150:
        return 0.3
    elif mean < 200:
        return 1.2
    else:
        return 1.5


def gamma_correction(img, gamma=None):
    normalized = img / 255.0

    if not gamma:
        gamma = get_gamma(img)

    print(img.std(), img.mean(), gamma)
    corrected = np.power(normalized, gamma)
    img = np.uint8(corrected * 255)
    return img, gamma

