import numpy as np
import cv2


class VisionCortex:
    def __init__(self):
        self.brightness_grid_size = 11  # → wektor 121-dim
        self.corners_grid_size = 6      # → wektor 36-dim

    def extract_features(self, gray_frame) -> dict:
        features = {}
        features["brightness"] = self._extract_brightness(gray_frame)
        features["corners"]    = self._extract_corners(gray_frame)
        return features

    # ------------------------------------------------------------------

    def _extract_brightness(self, gray_frame) -> np.ndarray:
        """
        Średnia jasność w siatce 11×11.
        Wynik: wektor (121,) w zakresie [0, 1].
        """
        h, w = gray_frame.shape
        g = self.brightness_grid_size
        cell_h, cell_w = h // g, w // g

        brightness = np.zeros((g, g))
        for i in range(g):
            for j in range(g):
                cell = gray_frame[
                    i * cell_h : (i + 1) * cell_h,
                    j * cell_w : (j + 1) * cell_w,
                ]
                brightness[i, j] = np.mean(cell) / 255.0

        return brightness.flatten()

    def _extract_corners(self, gray_frame) -> np.ndarray:
        """
        Gęstość narożników Harrisa w siatce 6×6.
        Wynik: wektor (36,) w zakresie [0, 1].

        Metoda:
          1. Harris corner response (blokSize=3, ksize=3, k=0.04)
          2. Próg: response > 1% maksimum → mapa binarna
          3. Liczenie aktywnych pikseli per komórkę → normalizacja przez rozmiar komórki
        """
        h, w = gray_frame.shape
        g = self.corners_grid_size
        cell_h, cell_w = h // g, w // g

        # Harris corner detection
        frame_f32 = np.float32(gray_frame)
        harris = cv2.cornerHarris(frame_f32, blockSize=3, ksize=3, k=0.04)

        # Binaryzacja — piksele silniejsze niż 1% maksimum uznajemy za narożniki
        threshold = 0.01 * harris.max() if harris.max() > 0 else 1.0
        corner_map = (harris > threshold).astype(np.float32)

        corners = np.zeros((g, g))
        for i in range(g):
            for j in range(g):
                cell = corner_map[
                    i * cell_h : (i + 1) * cell_h,
                    j * cell_w : (j + 1) * cell_w,
                ]
                # Normalizacja: gęstość narożników w komórce [0, 1]
                corners[i, j] = cell.mean()

        return corners.flatten()