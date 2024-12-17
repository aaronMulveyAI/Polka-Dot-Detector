import cv2
import numpy as np


class FeatureExtraction:

    @staticmethod
    def get_features(img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        mask = np.uint8(1 * (gray < threshold))

        B = (1 / 255) * np.sum(img[:, :, 0] * mask) / np.sum(mask)
        G = (1 / 255) * np.sum(img[:, :, 1] * mask) / np.sum(mask)
        R = (1 / 255) * np.sum(img[:, :, 2] * mask) / np.sum(mask)

        return [B, G, R]
