import cv2
import numpy as np


def binarize_global(img, thresh: int = 127, max_val: int = 255, method=cv2.THRESH_BINARY):
    """
    Global thresholding.

    Parameters:
    - img: Grayscale input image (uint8).
    - thresh: Fixed threshold value (0–255).
    - max_val: Maximum value assigned to pixels above threshold.
    - method: cv2.THRESH_BINARY or cv2.THRESH_BINARY_INV.

    Returns:
    - Binary image (uint8).
    """
    _, binary = cv2.threshold(img, thresh, max_val, method)
    return binary


def binarize_otsu(img, max_val: int = 255, method=cv2.THRESH_BINARY):
    """
    Otsu's automatic thresholding.

    Parameters:
    - img: Grayscale input image (uint8).
    - max_val: Maximum value assigned to pixels above threshold.
    - method: cv2.THRESH_BINARY or cv2.THRESH_BINARY_INV.

    Returns:
    - Binary image (uint8).
    """
    _, binary = cv2.threshold(img, 0, max_val, method + cv2.THRESH_OTSU)
    return binary


def binarize_adaptive_mean(img, max_val: int = 255, block_size: int = 35, C: int = 10):
    """
    Adaptive Mean thresholding.

    Parameters:
    - img: Grayscale input image (uint8).
    - max_val: Maximum value assigned to pixels above threshold.
    - block_size: Size of neighborhood (odd number > 1).
                  Larger values = smoother local mean.
    - C: Constant subtracted from the mean (fine-tunes threshold).

    Returns:
    - Binary image (uint8).
    """
    binary = cv2.adaptiveThreshold(img, max_val,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY,
                                   block_size, C)
    return binary


def binarize_adaptive_gaussian(img, max_val: int = 255, block_size: int = 35, C: int = 10):
    """
    Adaptive Gaussian thresholding.

    Parameters:
    - img: Grayscale input image (uint8).
    - max_val: Maximum value assigned to pixels above threshold.
    - block_size: Size of neighborhood (odd number > 1).
                  Larger values = smoother Gaussian-weighted mean.
    - C: Constant subtracted from the mean (fine-tunes threshold).

    Returns:
    - Binary image (uint8).
    """
    binary = cv2.adaptiveThreshold(img, max_val,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY,
                                   block_size, C)
    return binary


def binarize_sauvola(img, window_size: int = 25, k: float = 0.3, R: int = 128):
    """
    Sauvola local thresholding (robust for uneven illumination).

    Parameters:
    - img: Grayscale input image (uint8).
    - window_size: Neighborhood size for mean and std (odd number > 1).
    - k: Parameter in [0.2, 0.5], controls threshold sensitivity.
    - R: Dynamic range of standard deviation (default 128 for 8-bit images).

    Returns:
    - Binary image (uint8).
    """
    mean = cv2.boxFilter(img, cv2.CV_32F, (window_size, window_size))
    sqmean = cv2.boxFilter(img.astype(np.float32)**2, -1, (window_size, window_size))
    std = np.sqrt(np.maximum(sqmean - mean**2, 0))

    thresh = mean * (1 + k * (std / R - 1))
    binary = (img >= thresh).astype(np.uint8) * 255
    return binary


def binarize_niblack(img, window_size: int = 25, k: float = -0.2):
    """
    Niblack local thresholding.

    Parameters:
    - img: Grayscale input image (uint8).
    - window_size: Neighborhood size for mean and std (odd number > 1).
    - k: Negative value emphasizes darker text/objects on brighter background.

    Returns:
    - Binary image (uint8).
    """
    mean = cv2.boxFilter(img, cv2.CV_32F, (window_size, window_size))
    sqmean = cv2.boxFilter(img.astype(np.float32)**2, -1, (window_size, window_size))
    std = np.sqrt(np.maximum(sqmean - mean**2, 0))

    thresh = mean + k * std
    binary = (img >= thresh).astype(np.uint8) * 255
    return binary


def postprocess_binary(binary, min_size=200, kernel_size=3):
    # Morphological opening
    # Morphological opening + closing
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Connected components filtering
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    filtered = np.zeros_like(binary)
    for i in range(1, num_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_size:
            filtered[labels == i] = 255

    return filtered



def niblack_threshold(img, window_size=25, k=-0.2, min_size=100):
    """
    Apply Niblack thresholding using OpenCV and NumPy only.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image (uint8).
    window_size : int, optional
        Size of the local window (odd number, e.g., 15, 25).
    k : float, optional
        Niblack's k parameter, usually between -0.5 and 0.5.
    min_size : int, optional
        Minimum connected component area to keep (remove noise).

    Returns
    -------
    binary : np.ndarray
        Binary image (0 or 255).
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    gray = gray.astype(np.float32)

    # mean filter
    mean = cv2.boxFilter(gray, ddepth=-1, ksize=(window_size, window_size))

    # mean of squared values
    mean_sq = cv2.boxFilter(gray ** 2, ddepth=-1, ksize=(window_size, window_size))

    # variance = E[x^2] - (E[x])^2
    var = mean_sq - mean ** 2
    var[var < 0] = 0  # tránh sai số âm do float
    std = np.sqrt(var)

    # Niblack threshold
    thresh = mean + k * std
    binary = (gray > thresh).astype(np.uint8) * 255

    # remove small noise using connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    filtered = np.zeros_like(binary)
    for i in range(1, num_labels):  # skip background
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            filtered[labels == i] = 255

    return filtered



def hybrid_connected(gray, window_size=25, k=-0.2, min_size=300):
    # --- Step 1: Otsu
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- Step 2: Niblack
    niblack = niblack_threshold(gray, window_size=window_size, k=k, min_size=1)

    # --- Step 3: Connected components Otsu
    num_labels_otsu, labels_otsu, stats_otsu, _ = cv2.connectedComponentsWithStats(otsu, connectivity=8)
    final = np.zeros_like(otsu)

    # copy Otsu objects
    for i in range(1, num_labels_otsu):
        if stats_otsu[i, cv2.CC_STAT_AREA] >= min_size:
            final[labels_otsu == i] = 255

    # --- Step 4: Connected components Niblack (add missing ones)
    num_labels_nib, labels_nib, stats_nib, _ = cv2.connectedComponentsWithStats(niblack, connectivity=8)
    for j in range(1, num_labels_nib):
        if stats_nib[j, cv2.CC_STAT_AREA] < min_size:
            continue

        mask_nib = (labels_nib == j).astype(np.uint8) * 255
        overlap = cv2.bitwise_and(mask_nib, final)

        # nếu không overlap gì với Otsu → thêm vào (vì có thể là hạt gạo bị Otsu bỏ)
        if np.count_nonzero(overlap) == 0:
            final = cv2.bitwise_or(final, mask_nib)

    return final


def remove_small_components(binary, n_clusters=3):
    """
    Remove small white regions from a binary image.

    Parameters
    ----------
    binary : np.ndarray
        Input binary image (0/255).
    min_size : int
        Minimum area to keep (in pixels).

    Returns
    -------
    cleaned : np.ndarray
        Cleaned binary image.
    """
    # Ensure binary is 0/1
    img = (binary > 0).astype(np.uint8)

    # Connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
    if num_labels <= 1:
        return img * 255

    areas = stats[1:, cv2.CC_STAT_AREA]  # skip background
    labels_no_bg = labels.copy()
    labels_no_bg[labels_no_bg > 0] -= 1  # shift labels to 0..num_labels-2

    # --- Simple 1D clustering on areas ---
    areas_sorted = np.sort(areas)
    # split into n_clusters roughly equal-size bins
    bins = np.array_split(areas_sorted, n_clusters)
    cluster_means = [np.mean(b) for b in bins]

    # Assign each area to nearest cluster
    area_to_cluster = {}
    for i, a in enumerate(areas):
        distances = [abs(a - m) for m in cluster_means]
        cluster_id = np.argmin(distances)
        area_to_cluster[i] = cluster_id

    # Find smallest cluster
    cluster_sizes = {cid: 0 for cid in range(n_clusters)}
    for cid in area_to_cluster.values():
        cluster_sizes[cid] += 1
    smallest_cluster = min(cluster_sizes, key=cluster_sizes.get)

    # Remove objects in smallest cluster
    mask = np.zeros_like(img, dtype=np.uint8)
    for i in range(len(areas)):
        if area_to_cluster[i] != smallest_cluster:
            mask[labels_no_bg == i] = 1

    return (mask * 255).astype(np.uint8)