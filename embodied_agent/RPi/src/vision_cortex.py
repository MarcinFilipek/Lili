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
