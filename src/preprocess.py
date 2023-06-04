import cv2
import numpy as np


def extract_from_adaptive_threshold(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    thresholded = cv2.adaptiveThreshold(
        cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    thresholded_median = cv2.medianBlur(thresholded, 5)
    inv_thresholded_median = cv2.bitwise_not(thresholded_median)
    return inv_thresholded_median.astype(np.float32) / 255


def append_threshold_channel(image: np.ndarray) -> np.ndarray:
    thresholded = cv2.adaptiveThreshold(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    extended = np.dstack((image, thresholded))
    return extended.astype(np.float32) / 127.5 - 1
