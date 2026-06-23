import numpy as np
import cv2


class VisionCortex:
    def __init__(self):
        self.grid_size = 11  # → wektory 121-dim

    def extract_features(self, gray_frame) -> dict:
        """
        Args:
            gray_frame: surowa klatka w skali szarości (bez blura)
        """
        blurred = self._blur(gray_frame)
        return {
            "brightness": self._extract_brightness(blurred),
            "edges":      self._extract_edges(gray_frame),  # surowy obraz
        }

    def _blur(self, frame) -> np.ndarray:
        """pyrDown×2 + pyrUp×2 — wygładza szum, zachowuje globalne struktury."""
        return cv2.pyrUp(cv2.pyrUp(cv2.pyrDown(cv2.pyrDown(frame))))

    def _extract_brightness(self, gray_frame) -> np.ndarray:
        """Średnia jasność w siatce 11×11 na blurowanym obrazie. Wynik: (121,) w [0, 1]."""
        h, w = gray_frame.shape
        g = self.grid_size
        cell_h, cell_w = h // g, w // g
        grid = np.zeros((g, g))
        for i in range(g):
            for j in range(g):
                cell = gray_frame[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                grid[i, j] = np.mean(cell) / 255.0
        return grid.flatten()

    def _extract_edges(self, gray_frame) -> np.ndarray:
        """
        Gęstość krawędzi Sobela w siatce 11×11 na surowym obrazie. Wynik: (121,) w [0, 1].

        Metoda:
          1. Sobel X + Sobel Y → magnituda gradientu
          2. Normalizacja globalna do [0, 1]
          3. Średnia magnituda per komórka siatki
        """
        sobelx = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)

        max_val = magnitude.max()
        if max_val > 0:
            magnitude = magnitude / max_val

        h, w = magnitude.shape
        g = self.grid_size
        cell_h, cell_w = h // g, w // g
        grid = np.zeros((g, g))
        for i in range(g):
            for j in range(g):
                cell = magnitude[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                grid[i, j] = cell.mean()
        return grid.flatten()