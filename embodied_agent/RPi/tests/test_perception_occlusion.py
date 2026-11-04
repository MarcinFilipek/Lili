from pathlib import Path

import cv2
import numpy as np
import pytest

from lili.body.camera_model import CameraModel
from lili.body.efference import Efference
from lili.body.geometry import Pose
from lili.perception.occlusion import drop_small_regions, fill_far, grow, occlusion_edges
from lili.pipeline import process_pair
from synthetic import PINHOLE, boundaries, edge_scores, ground_truth, render, table_scene

CALIB_DIR = Path(__file__).resolve().parent.parent / "calib"
SCENE = table_scene()


def test_fill_far_fills_band_with_background():
    values = np.zeros((5, 30), np.float32)
    values[:, :10] = 2.0  # bliski obiekt po lewej, tło (0) po prawej
    reliable = np.ones((5, 30), bool)
    reliable[:, 10:20] = False  # pas zasłonięcia
    filled, supported = fill_far(values, reliable)
    assert supported.all()
    np.testing.assert_array_equal(filled[:, 10:20], 0)
    np.testing.assert_array_equal(filled[:, :10], 2)


def test_fill_far_closes_large_holes():
    reliable = np.zeros((40, 40), bool)
    reliable[0, 0] = True
    filled, supported = fill_far(np.full((40, 40), 1.5, np.float32), reliable, radius=3)
    np.testing.assert_array_equal(filled, 1.5)
    assert supported[:4, :4].all() and not supported[4:, 4:].any()


def test_grow_extends_known_values():
    known = np.zeros((1, 10), bool)
    known[0, :3] = True
    values, grown = grow(np.full((1, 10), 4.0), known, 2)
    assert grown[0].tolist() == [True] * 5 + [False] * 5
    np.testing.assert_array_equal(values[0, :5], 4.0)


def test_drop_small_regions():
    mask = np.zeros((20, 20), bool)
    mask[:10, :10] = True  # 100 px
    mask[15, 15] = True  # wysepka
    kept = drop_small_regions(mask, 50)
    assert kept[:10, :10].all() and not kept[15, 15]


def test_edges_of_true_depth_follow_object_outlines():
    truth = ground_truth(SCENE, PINHOLE)
    gain = PINHOLE.fx * 0.02  # przesuw 2 cm w bok
    occlusion = occlusion_edges(1 / truth.depth, np.ones(truth.depth.shape, bool), gain)
    precision, recall = edge_scores(occlusion.edges, boundaries(truth.ids), np.ones_like(occlusion.edges), 1)
    assert precision > 0.95 and recall > 0.95
    # Na prawej krawędzi obiektu 2 (0.55 m) w wierszu 120 zaczyna się obiekt 3 (0.8 m):
    # bliżej jest lewa strona, skok paralaksy = -(1/0.55 - 1/0.8) * gain [px]
    right = round(PINHOLE.cx + PINHOLE.fx * (0.06 + 0.07) / 0.55)
    expected = -(1 / 0.55 - 1 / 0.8) * gain
    assert occlusion.jump_x[120, right - 3: right + 3].min() == pytest.approx(expected, rel=0.15)


def test_constant_rho_offset_keeps_edges():
    """Drobny obrót dodaje do rho prawie stałą: krawędzie (różnice paralaksy) mają się nie zmienić."""
    truth = ground_truth(SCENE, PINHOLE)
    reliable = np.ones(truth.depth.shape, bool)
    gain = PINHOLE.fx * 0.02
    clean = occlusion_edges(1 / truth.depth, reliable, gain)
    shifted = occlusion_edges(1 / truth.depth + 0.5, reliable, gain)
    np.testing.assert_array_equal(clean.edges, shifted.edges)


@pytest.mark.parametrize("name, cmd_b, T", [
    ("peering", (10, 0), None),
    ("peering", (-10, 0), None),
    ("reka", None, [0.02, 0, 0]),
    ("reka", None, [-0.02, 0, 0]),
])
def test_chain_finds_object_outlines(name, cmd_b, T):
    """Kryterium M4 na scenach syntetycznych: wykryte krawędzie leżą na konturach obiektów."""
    camera = CameraModel.load(CALIB_DIR / "camera_320x240.json")
    if T is None:
        pose = Efference.from_calibration(CALIB_DIR).motion((0, 0), cmd_b)
    else:
        pose = Pose(np.eye(3), np.array(T))
    result = process_pair(render(SCENE, camera), render(SCENE, camera, pose), camera, pose)

    truth = ground_truth(SCENE, CameraModel(camera.width, camera.height, camera.K, np.zeros(5)))
    inner = cv2.erode(result.valid.astype(np.uint8), np.ones((9, 9))).astype(bool)
    edges_true = boundaries(truth.ids)
    precision, _ = edge_scores(result.edges, edges_true, inner)
    assert precision > 0.8
    # Obiekty 2 i 3 w środku kadru; obiekt 1 (0.35 m, przy lewym brzegu) DIS często gubi
    for i in (2, 3):
        near_object = cv2.dilate((truth.ids == i).astype(np.uint8), np.ones((3, 3))).astype(bool)
        _, recall = edge_scores(result.edges, edges_true & near_object, inner)
        assert recall > 0.7, f"obiekt {i}"


def test_steep_ramp_is_not_an_edge():
    """Podłoga tuż przed kamerą: rho rośnie szybko, ale płynnie w dół obrazu. Skok na niej ma zostać."""
    rows = np.arange(240, dtype=float)[:, None] * np.ones((1, 320))
    rho = 1 + 0.2 * rows  # przy gain 7.5 to 1.5 px paralaksy na wiersz
    rho[100:140, 150:200] += 3  # bliski obiekt na podłodze
    occlusion = occlusion_edges(rho, np.ones(rho.shape, bool), 7.5)
    edges = occlusion.edges
    assert not edges[20:80].any() and not edges[180:].any()
    assert edges[95:145, 145:205].sum() > 100


def test_textureless_unknown_gives_no_edges():
    """Niepewna plama bez tekstury to niewiadoma, nie zasłonięte tło: brak krawędzi wokół pewnej wysepki."""
    rho = np.full((120, 160), 0.7)  # daleka tkanina u góry
    reliable = np.zeros(rho.shape, bool)
    reliable[:60] = True
    rho[70:90, 70:90] = 3.0  # wysepka bliskiej podłogi z odrobiną tekstury, 10 px pod tkaniną
    reliable[70:90, 70:90] = True
    fillable = reliable.copy()  # reszta podłogi bez tekstury
    edges = occlusion_edges(rho, reliable, 7.5, fillable=fillable).edges
    assert not edges[55:].any()
    # Bez informacji o teksturze wysepka dostaje kontur (tło wypełnione tkaniną)
    assert occlusion_edges(rho, reliable, 7.5).edges[60:100, 60:100].any()
