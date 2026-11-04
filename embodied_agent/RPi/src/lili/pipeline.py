"""Percepcja dla pary widoków A -> B: z sesji (lub dowolnych obrazów i ruchu) do 1/Z i krawędzi zasłaniania."""

from dataclasses import dataclass

import cv2
import numpy as np

from lili.body.geometry import Pose
from lili.perception.derotation import derotate_pair
from lili.perception.flow import dense_flow, forward_backward_error, photometric_error, texture_strength
from lili.perception.occlusion import occlusion_edges
from lili.perception.parallax import estimate_translation, parallax, rho_sigma
from lili.record import schema as s


@dataclass
class View:
    """Postój ruch-stop-patrz: uśredniony obraz i stan ciała."""

    step: int
    image: np.ndarray  # mediana klatek fazy look (mniej szumu sensora)
    pan: float  # komenda [jednostki protokołu]
    tilt: float
    offset: np.ndarray  # deklarowane położenie podstawy [m], zera gdy się nie ruszała
    frames: int


def views(session):
    """Widoki z faz look sesji, w kolejności kroków."""
    frames = session.load_frames()
    events = session.events()
    phases = [e for e in events if e["type"] == "phase"]
    result = []
    for i, event in enumerate(phases):
        if event["phase"] != s.PHASE_LOOK:
            continue
        t_end = phases[i + 1]["t"] if i + 1 < len(phases) else np.inf
        before = [e for e in events if e["t"] <= event["t"]]
        cmd = next(e for e in reversed(before) if e["type"] == "cmd")
        base = next((e for e in reversed(before) if e["type"] == "base"), None)
        selected = (frames.t >= event["t"]) & (frames.t < t_end)
        if not selected.any():
            continue
        result.append(View(
            step=event["step"],
            image=np.median(frames.images[selected], axis=0).round().astype(np.uint8),
            pan=cmd["pan"],
            tilt=cmd["tilt"],
            offset=np.zeros(3) if base is None else np.array(base["offset"], float),
            frames=int(selected.sum()),
        ))
    return result


def motion(view_a, view_b, efference):
    """Ruch kamery A -> B w układzie kamery A: głowa z kopii eferentnej plus deklarowane przesunięcie podstawy."""
    def pose(view):
        head = efference.pose(view.pan, view.tilt)
        return Pose(head.R, head.c + view.offset)

    return pose(view_b).relative_to(pose(view_a))


@dataclass
class PairResult:
    a: np.ndarray  # A po korekcji dystorsji
    b: np.ndarray  # B po korekcji dystorsji i obrotu
    valid: np.ndarray  # wspólne pole widzenia
    flow: np.ndarray  # A -> B'
    T: np.ndarray  # użyte przesunięcie kamery [m]
    rho: np.ndarray  # 1/Z [1/m]
    sigma: np.ndarray  # niepewność rho [1/m]
    reliable: np.ndarray  # piksele z wiarygodnym rho
    edges: np.ndarray  # krawędzie zasłaniania


def reliable_mask(rho, sigma, photo, texture, valid, max_rel_sigma=0.25, max_photo=8.0, min_texture=1.0):
    """Piksele z wiarygodnym rho: mała niepewność względna, zgodny obraz po przeniesieniu przepływem
    i dość tekstury w A.

    Błąd fotometryczny odrzuca tło zasłonięte w B, któremu DIS przypisał ruch obiektu. Na plamach
    bez tekstury (podłoga, gładka ściana) DIS interpoluje przepływ z sąsiedztwa, a błąd
    fotometryczny jest mały przy każdym przepływie, więc potrzebny jest osobny próg tekstury.
    Szum sensora daje ok. 0.2 poziomu / px.
    """
    with np.errstate(invalid="ignore"):
        return valid & (sigma < max_rel_sigma * np.abs(rho)) & (photo < max_photo) & (texture > min_texture)


def process_pair(image_a, image_b, camera, pose, refine_translation=False):
    """Pełny łańcuch dla pary obrazów i ruchu kamery pose (R, T) z A do B."""
    a, b, valid = derotate_pair(image_a, image_b, camera, pose.R)
    flow, back = dense_flow(a, b), dense_flow(b, a)
    fb = forward_backward_error(flow, back)
    T = pose.c
    if refine_translation:
        # Przy ręcznym przesuwaniu kierunek jest znany tylko w przybliżeniu
        T = estimate_translation(flow, camera.K, T, valid & (fb < 0.5))
    par = parallax(flow, camera.K, T)
    sigma = rho_sigma(par, fb_error=fb)
    reliable = reliable_mask(par.rho, sigma, photometric_error(a, b, flow), texture_strength(a), valid)
    occlusion = occlusion_edges(par.rho, reliable)
    # Przy brzegu wspólnego pola widzenia wypełnienie poza nim daje sztuczne skoki
    inner = cv2.erode(valid.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    return PairResult(a, b, valid, flow, T, par.rho, sigma, occlusion.reliable, occlusion.edges & inner)
