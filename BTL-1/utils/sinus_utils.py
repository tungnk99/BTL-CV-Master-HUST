import numpy as np


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