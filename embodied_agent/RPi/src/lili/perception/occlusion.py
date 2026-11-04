"""Krawędzie zasłaniania ze skoków odwrotnej głębi.

Skok mierzony jest jako różnica rho przeliczona na piksele przepływu (razy gain z parallax), czyli
jako różnica paralaksy obu stron krawędzi. Taki próg porównuje się wprost z szumem przepływu,
sam skaluje się z długością T, a stały dodatek do rho od nieznanego drobnego obrotu kamery
(ok. 0.3 deg przy ręcznym przesuwaniu) znosi się w różnicy. Stosunek odległości (log rho) nie ma
tej własności: dodatek go zmniejsza, najbardziej dla dalekich obiektów.

DIS rozmywa przepływ na krawędzi na kilka pikseli, a tuż przy niej przepływ jest niepewny
(odsłonięcia), więc niepewne piksele wypełniane są dalszym z pewnych sąsiadów, a skok mierzy
gradient wygładzonego rho. Cienkie krawędzie daje tłumienie niemaksymalnych wartości z cv2.Canny.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Occlusion:
    edges: np.ndarray  # (H, W) bool, cienkie krawędzie zasłaniania
    jump_x: np.ndarray  # (H, W) skok paralaksy [px] w poprzek kolumn; > 0: bliżej jest prawa strona
    jump_y: np.ndarray  # (H, W) skok paralaksy [px] w poprzek wierszy; > 0: bliżej jest dolna strona
    reliable: np.ndarray  # (H, W) bool, piksele z wiarygodnym rho
    supported: np.ndarray  # (H, W) bool, pewne lub z pewnym pikselem w promieniu wypełniania


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


def fill_far(rho, reliable, radius=15, fillable=None):
    """Wypełnia niepewne piksele najmniejszym rho (najdalszym) z pewnych pikseli w promieniu radius.

    Niepewne pasy przy krawędziach to tło zasłaniane lub odsłaniane przez bliższy obiekt
    (sam obiekt widać w obu ujęciach), więc wypełnienie tłem stawia skok na konturze obiektu.
    radius powinien obejmować szerokość pasa, czyli różnicę przepływu obiektu i tła.
    Większe dziury wypełniane są dalej, krok po kroku, ale tam wartości z różnych stron stykają
    się w przypadkowych liniach, więc zwracana jest też maska supported: piksele pewne lub
    z pewnym pikselem w promieniu radius.
    fillable: piksele, które wolno wypełnić tłem z promienia radius (domyślnie wszystkie). Plamy bez
    tekstury nie są zasłonięciem, tylko niewiadomą: wypełniane są dopiero krok po kroku i nie
    liczą się jako supported.
    """
    unknown = 1e30  # cv2.erode zamienia inf na największy float32, więc zwykły duży znacznik
    filled = np.where(reliable, rho, unknown).astype(np.float32)
    window = cv2.erode(filled, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
    if fillable is not None:
        window[~fillable] = unknown
    filled = np.where(reliable, filled, window)
    known = filled < unknown
    supported = known.copy()
    kernel = np.ones((3, 3), np.uint8)
    while not known.all():
        grown = cv2.erode(filled, kernel)
        new = ~known & (grown < unknown)
        if not new.any():
            filled[~known] = 0  # brak jakiegokolwiek pewnego piksela
            break
        filled[new] = grown[new]
        known |= new
    return filled, supported


def drop_small_regions(mask, min_area):
    """Usuwa spójne obszary maski mniejsze niż min_area pikseli (wysepki w pasach zasłonięć)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False  # tło maski
    return keep[labels]


def occlusion_edges(rho, reliable, gain, fillable=None, min_jump_px=1.5, smooth=1.5, trend=6.0, step_ratio=2.0,
                    min_area=50, near_grow=2):
    """Krawędzie, na których paralaksa skacze o więcej niż min_jump_px pikseli.

    reliable: piksele z wiarygodnym rho (np. pipeline.reliable_mask), reszta wypełniana tłem.
    gain: przepływ [px] na jednostkę rho w każdym pikselu (Parallax.gain).
    fillable: niepewne piksele, które mogą być zasłoniętym tłem (z teksturą); patrz fill_far.
    smooth: sigma wygładzenia rho [px] przed liczeniem gradientu.
    trend: sigma [px] szerszej skali do odróżnienia skoku od rampy. Płaszczyzna pochylona względem
    kamery (podłoga tuż przed nią) ma stromy, ale stały gradient rho: w obu skalach taki sam.
    Przy skoku gradient w skali smooth jest trend / smooth razy większy niż w skali trend,
    więc krawędź wymaga co najmniej step_ratio razy większego.
    """
    reliable = drop_small_regions(reliable & np.isfinite(rho) & (rho > 0), min_area)
    # DIS rozmywa przepływ także na kilku pikselach obiektu przy krawędzi: te odzyskuje
    # krótkie rozszerzenie pewnych wartości, dopiero resztę pasa wypełnia tło
    filled, known = grow(np.where(reliable, rho, 0.0), reliable, near_grow)
    filled, supported = fill_far(filled, known, fillable=None if fillable is None else fillable | known)
    filled = cv2.medianBlur(filled, 3)
    fine_y, fine_x = np.gradient(cv2.GaussianBlur(filled, (0, 0), smooth))
    coarse_y, coarse_x = np.gradient(cv2.GaussianBlur(filled, (0, 0), trend))
    is_step = np.hypot(fine_x, fine_y) > step_ratio * np.hypot(coarse_x, coarse_y)

    # Gradient przeskalowany tak, że idealny skok o J daje w szczycie J, potem na piksele przepływu
    scale = np.sqrt(2 * np.pi) * smooth
    jump_x, jump_y = fine_x * scale * gain, fine_y * scale * gain

    # Canny na gotowych pochodnych: tłumienie niemaksymalnych i histereza (słabsze krawędzie
    # przyłączone do mocnych), w jednostkach 1/100 px
    dx = np.clip(jump_x * 100, -32767, 32767).astype(np.int16)
    dy = np.clip(jump_y * 100, -32767, 32767).astype(np.int16)
    edges = cv2.Canny(dx, dy, 0.6 * min_jump_px * 100, min_jump_px * 100, L2gradient=True) > 0
    # Bez krawędzi tam, gdzie wypełnienie nie ma oparcia w pobliskich pewnych pikselach (duże plamy
    # bez tekstury); margines na wygładzenie
    supported_inner = cv2.erode(supported.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    return Occlusion(edges & supported_inner & is_step, jump_x, jump_y, reliable, supported)
