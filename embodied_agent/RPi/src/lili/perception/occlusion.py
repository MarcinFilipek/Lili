"""Krawędzie zasłaniania ze skoków odwrotnej głębi.

Skok mierzony jest w log(rho): różnica log(rho) to stosunek odległości, więc ten sam próg działa
dla obiektu na tle 1 m i 3 m. DIS rozmywa przepływ na krawędzi na kilka pikseli, a tuż przy niej
przepływ jest niepewny (odsłonięcia), więc niepewne piksele wypełniane są dalszym z pewnych
sąsiadów, a skok mierzy gradient wygładzonego log(rho). Cienkie krawędzie daje tłumienie
niemaksymalnych wartości z cv2.Canny.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Occlusion:
    edges: np.ndarray  # (H, W) bool, cienkie krawędzie zasłaniania
    jump_x: np.ndarray  # (H, W) skok log(rho) w poprzek kolumn; > 0: bliżej jest prawa strona
    jump_y: np.ndarray  # (H, W) skok log(rho) w poprzek wierszy; > 0: bliżej jest dolna strona
    reliable: np.ndarray  # (H, W) bool, piksele z wiarygodnym rho


def grow(values, known, steps):
    """Rozszerza znane wartości o steps pikseli: nowy piksel = średnia znanych sąsiadów 3x3."""
    values = np.where(known, values, 0).astype(np.float32)
    known = known.copy()
    for _ in range(steps):
        total = cv2.boxFilter(values, -1, (3, 3), normalize=False)
        count = cv2.boxFilter(known.astype(np.float32), -1, (3, 3), normalize=False)
        new = ~known & (count > 0)
        values[new] = total[new] / count[new]
        known |= new
    return values, known


def fill_far(log_rho, reliable, radius=15):
    """Wypełnia niepewne piksele najmniejszym log(rho) (najdalszym) z pewnych pikseli w promieniu radius.

    Niepewne pasy przy krawędziach to tło zasłaniane lub odsłaniane przez bliższy obiekt
    (sam obiekt widać w obu ujęciach), więc wypełnienie tłem stawia skok na konturze obiektu.
    radius powinien obejmować szerokość pasa, czyli różnicę przepływu obiektu i tła.
    Większe dziury wypełniane są dalej, krok po kroku.
    """
    unknown = 1e30  # cv2.erode zamienia inf na największy float32, więc zwykły duży znacznik
    filled = np.where(reliable, log_rho, unknown).astype(np.float32)
    window = cv2.erode(filled, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
    filled = np.where(reliable, filled, window)
    known = filled < unknown
    kernel = np.ones((3, 3), np.uint8)
    while not known.all():
        grown = cv2.erode(filled, kernel)
        new = ~known & (grown < unknown)
        if not new.any():
            filled[~known] = 0  # brak jakiegokolwiek pewnego piksela
            break
        filled[new] = grown[new]
        known |= new
    return filled


def drop_small_regions(mask, min_area):
    """Usuwa spójne obszary maski mniejsze niż min_area pikseli (wysepki w pasach zasłonięć)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False  # tło maski
    return keep[labels]


def occlusion_edges(rho, reliable, jump=0.3, smooth=1.5, min_area=50, near_grow=2):
    """Krawędzie, na których log(rho) skacze o więcej niż jump (0.3 = odległości różne o ok. 35%).

    reliable: piksele z wiarygodnym rho (np. pipeline.reliable_mask), reszta wypełniana tłem.
    smooth: sigma wygładzenia log(rho) [px] przed liczeniem gradientu.
    """
    reliable = drop_small_regions(reliable & np.isfinite(rho) & (rho > 0), min_area)
    # DIS rozmywa przepływ także na kilku pikselach obiektu przy krawędzi: te odzyskuje
    # krótkie rozszerzenie pewnych wartości, dopiero resztę pasa wypełnia tło
    log_rho, known = grow(np.log(np.where(reliable, rho, 1.0)), reliable, near_grow)
    log_rho = fill_far(log_rho, known)
    log_rho = cv2.GaussianBlur(cv2.medianBlur(log_rho, 3), (0, 0), smooth)

    # Gradient przeskalowany tak, że idealny skok o J daje w szczycie J
    scale = np.sqrt(2 * np.pi) * smooth
    jump_y, jump_x = (g * scale for g in np.gradient(log_rho))

    # Canny na gotowych pochodnych: tłumienie niemaksymalnych i histereza (słabsze krawędzie
    # przyłączone do mocnych), w jednostkach 1/1000 log(rho)
    dx = np.clip(jump_x * 1000, -32767, 32767).astype(np.int16)
    dy = np.clip(jump_y * 1000, -32767, 32767).astype(np.int16)
    edges = cv2.Canny(dx, dy, 0.6 * jump * 1000, jump * 1000, L2gradient=True) > 0
    return Occlusion(edges, jump_x, jump_y, reliable)
