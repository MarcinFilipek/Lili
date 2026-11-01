"""Gęsty przepływ optyczny: jeden interfejs, wymienna metoda.

Przepływ (H, W, 2) float32 w pikselach: piksel (x, y) obrazu a trafia w (x, y) + flow[y, x] obrazu b.
"""

import cv2
import numpy as np


def dis_flow(a, b, preset=cv2.DISOPTICAL_FLOW_PRESET_MEDIUM):
    """Dense Inverse Search (OpenCV): szybki na RPi, dokładny subpikselowo na teksturze."""
    dis = cv2.DISOpticalFlow_create(preset)
    return dis.calc(np.ascontiguousarray(a), np.ascontiguousarray(b), None)


def farneback_flow(a, b):
    return cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 15, 5, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)


METHODS = {"dis": dis_flow, "farneback": farneback_flow}


def dense_flow(a, b, method="dis"):
    """Przepływ a -> b dla obrazów uint8 w skali szarości."""
    return METHODS[method](a, b).astype(np.float32)


def warp_by_flow(image, flow):
    """Wartości image w punktach (x, y) + flow: przenosi obraz b (lub przepływ b -> a) na siatkę a."""
    h, w = flow.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(image, xs + flow[..., 0], ys + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def forward_backward_error(flow_ab, flow_ba):
    """|flow_ab(x) + flow_ba(x + flow_ab(x))| [px]: duży przy zasłonięciach i błędach dopasowania."""
    return np.linalg.norm(flow_ab + warp_by_flow(flow_ba, flow_ab), axis=-1)


def photometric_error(a, b, flow, window=5):
    """Średnia |a - b przeniesione przepływem| w oknie: duża tam, gdzie przepływ wskazuje inną treść
    (np. tło zasłonięte w b, któremu przepływ przypisał ruch obiektu).
    """
    warped = warp_by_flow(b.astype(np.float32), flow)
    return cv2.blur(np.abs(warped - a.astype(np.float32)), (window, window))
