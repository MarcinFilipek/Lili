import numpy as np
import cv2


class VisionCortex:
    def __init__(self):
        self.feature_dims = {
            "brightness": 16,  # 4×4 grid
            "edges": 64,  # 8×8 grid
            "motion": 32,  # 4×4 flow vectors (dx, dy)
            "corners": 36,  # 6×6 grid
        }

    def extract_features(self, gray_frame):
        features = {}
        features["brightness"] = self._extract_brightness(gray_frame)
        features["corners"] = self._extract_corners(gray_frame)

        print(f"Brightness: {features['brightness']}")
        print(f"Corners: {features['corners']}")
        return features

    def _extract_brightness(self, gray_frame):
        """Brightness in 4×4 grid"""
        h, w = gray_frame.shape
        grid_size = 4
        cell_h, cell_w = h // grid_size, w // grid_size

        brightness = np.zeros((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                cell = gray_frame[
                    i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w
                ]
                brightness[i, j] = np.mean(cell) / 255.0

        return brightness.flatten()

    def _extract_corners(self, gray_frame):
        """
        Extract corners in 6×6 grid using Harris corner detector.

        Args:
            gray_frame: grayscale image (H, W)

        Returns:
            corners: np.array([36]) - flattened 6x6 grid
        """
        # Convert to float32 for Harris
        gray_float = np.float32(gray_frame)

        # Harris corner detection
        corners = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)

        # Normalize
        if corners.max() > 0:
            corners = corners / corners.max()

        # Resize to 6×6
        corners_small = cv2.resize(corners, (6, 6), interpolation=cv2.INTER_AREA)

        return corners_small.flatten()  # shape: (36,)
