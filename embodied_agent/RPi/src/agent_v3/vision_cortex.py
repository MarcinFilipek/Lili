import numpy as np


class VisionCortex:
    def __init__(self):
        self.brightness_grid_size = 3

    def extract_features(self, gray_frame):
        features = {}
        features["brightness"] = self._extract_brightness(gray_frame)
        return features

    def _extract_brightness(self, gray_frame):
        """Brightness in 3×3 grid (brightness_grid_size=3 → 9-dim vector)"""
        h, w = gray_frame.shape
        g = self.brightness_grid_size
        cell_h, cell_w = h // g, w // g

        brightness = np.zeros((g, g))
        for i in range(g):
            for j in range(g):
                cell = gray_frame[
                    i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w
                ]
                brightness[i, j] = np.mean(cell) / 255.0

        return brightness.flatten()
