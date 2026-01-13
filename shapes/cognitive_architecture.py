# cognitive_architecture.py

import torch
import torch.nn as nn
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# ============================================================================
# MODULE 1: EARLY VISUAL CORTEX (V1/V2)
# ============================================================================


class EarlyVisualCortex(nn.Module):
    """
    Symuluje V1/V2 - wykrywa krawędzie, orientacje, kontrast.

    Biologicznie:
    - Gabor-like receptive fields
    - Orientation columns
    - Edge detection
    """

    def __init__(self):
        super().__init__()

        # Simple cells (wykrywają krawędzie w różnych orientacjach)
        self.simple_cells = nn.ModuleList(
            [self._create_gabor_filter(angle) for angle in [0, 45, 90, 135]]
        )

        # Complex cells (niezmiennicze na pozycję)
        self.complex_cells = nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=2)
        self.pool = nn.MaxPool2d(2, 2)

    def _create_gabor_filter(self, angle):
        """Gabor filter dla danej orientacji."""
        gabor = nn.Conv2d(3, 4, kernel_size=11, padding=5, bias=False)

        # Inicjalizuj wagi jako Gabor
        with torch.no_grad():
            for i in range(4):
                kernel = self._gabor_kernel(11, angle + i * 45, frequency=0.1)
                gabor.weight[i, :, :, :] = (
                    torch.tensor(kernel).unsqueeze(0).repeat(3, 1, 1)
                )

        return gabor

    def _gabor_kernel(self, size, theta, frequency):
        """Tworzy Gabor kernel."""
        sigma = size / 4.0
        lambd = 1.0 / frequency
        gamma = 0.5
        psi = 0

        theta_rad = np.radians(theta)

        center = size // 2
        x = np.arange(size) - center
        y = np.arange(size) - center
        X, Y = np.meshgrid(x, y)

        x_theta = X * np.cos(theta_rad) + Y * np.sin(theta_rad)
        y_theta = -X * np.sin(theta_rad) + Y * np.cos(theta_rad)

        gb = np.exp(
            -0.5 * (x_theta**2 / sigma**2 + y_theta**2 / (sigma / gamma) ** 2)
        ) * np.cos(2 * np.pi * x_theta / lambd + psi)

        return gb.astype(np.float32)

    def forward(self, x):
        """
        Input: [batch, 3, 512, 512]
        Output: edge_map [batch, 16, 128, 128]
        """
        # Simple cells - różne orientacje
        orientations = []
        for cell in self.simple_cells:
            orientations.append(cell(x))

        # Połącz orientacje
        orientation_map = torch.cat(orientations, dim=1)  # [batch, 16, 512, 512]

        # Complex cells - integracja
        edge_features = self.complex_cells(orientation_map)  # [batch, 16, 256, 256]
        edge_features = self.pool(edge_features)  # [batch, 16, 128, 128]

        return edge_features


# ============================================================================
# MODULE 2: SHAPE PROCESSING (V4/IT)
# ============================================================================


class ShapeProcessor(nn.Module):
    """
    Symuluje V4/IT - wykrywa rogi, kontury, części kształtów.

    Biologicznie:
    - Corner detection cells
    - Contour integration
    - Part-based representation
    """

    def __init__(self):
        super().__init__()

        # Corner detector (wykrywa rogi)
        self.corner_detector = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
        )

        # Corner localization (gdzie są rogi?)
        self.corner_heatmap = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1), nn.Sigmoid()
        )

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))

    def forward(self, edge_features):
        """
        Input: edge_features [batch, 16, 128, 128]
        Output:
            - corner_features [batch, 64, 128, 128]
            - corner_heatmap [batch, 1, 128, 128]
            - global_features [batch, 64, 4, 4]
        """
        # Wykryj rogi
        corner_features = self.corner_detector(edge_features)

        # Gdzie są rogi? (heatmap)
        corner_heatmap = self.corner_heatmap(corner_features)

        # Global representation
        global_features = self.global_pool(corner_features)

        return corner_features, corner_heatmap, global_features


# ============================================================================
# MODULE 3: NUMBER SENSE (IPS - Intraparietal Sulcus)
# ============================================================================


class NumberSense(nn.Module):
    """
    Symuluje IPS - wrodzona zdolność do "subitizing" (szybkie liczenie 1-4).

    Biologicznie:
    - Działa automatycznie, nieświadomie
    - Dokładne dla 1-4, przybliżone dla więcej
    - Aktywuje się przy wykrywaniu "ilości"
    """

    def __init__(self):
        super().__init__()

        # Approximate Number System (ANS)
        # Wykrywa "ile" bez precyzyjnego liczenia
        self.numerosity_detector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 10),  # 0-9 obiektów (ale rozmazane dla >4)
        )

        # Confidence dla każdej liczby
        self.confidence = nn.Softmax(dim=1)

    def forward(self, global_features, corner_heatmap):
        """
        Input:
            - global_features [batch, 64, 4, 4]
            - corner_heatmap [batch, 1, 128, 128]

        Output:
            - numerosity_estimate [batch, 10] (prawdopodobieństwa 0-9)
            - exact_count (wykryte z heatmap)
        """
        # ANS estimate (neuronal)
        numerosity_logits = self.numerosity_detector(global_features)
        numerosity_probs = self.confidence(numerosity_logits)

        # Exact count z heatmap (symboliczny)
        # Znajdź local maxima w heatmap
        exact_count = self._count_peaks(corner_heatmap)

        return numerosity_probs, exact_count

    def _count_peaks(self, heatmap):
        """
        Policzy peaks w heatmap (miejsca gdzie są rogi).
        To jest symboliczna operacja (nie gradient).
        """
        batch_size = heatmap.shape[0]
        counts = []

        for b in range(batch_size):
            # Konwertuj do numpy
            heat = heatmap[b, 0].detach().cpu().numpy()

            # Threshold
            peaks = heat > 0.5

            # Znajdź connected components (każdy to jeden róg)
            from scipy import ndimage

            labeled, num_features = ndimage.label(peaks)

            counts.append(num_features)

        return torch.tensor(counts, device=heatmap.device)


# ============================================================================
# MODULE 4: WORKING MEMORY (Prefrontal Cortex)
# ============================================================================


class WorkingMemory:
    """
    Symuluje working memory - trzyma aktualne informacje "w świadomości".

    Biologicznie:
    - Ograniczona pojemność (~4-7 items)
    - Utrzymuje informację przez krótki czas
    - Może być aktualizowana przez attention

    To NIE jest neural module - to buffer/pamięć!
    """

    def __init__(self, capacity=7):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.attention_weights = []

    def update(self, item, importance=1.0):
        """
        Dodaj item do working memory.
        Jeśli pełna - usuń najmniej ważny.
        """
        if len(self.buffer) >= self.capacity:
            # Usuń item z najniższą uwagą
            min_idx = np.argmin(self.attention_weights)
            del self.buffer[min_idx]
            del self.attention_weights[min_idx]

        self.buffer.append(item)
        self.attention_weights.append(importance)

    def read(self):
        """Odczytaj wszystko z working memory."""
        return list(self.buffer)

    def clear(self):
        """Wyczyść working memory."""
        self.buffer.clear()
        self.attention_weights.clear()

    def get_most_salient(self):
        """Zwróć najbardziej ważny item."""
        if not self.buffer:
            return None
        max_idx = np.argmax(self.attention_weights)
        return self.buffer[max_idx]


# ============================================================================
# MODULE 5: SEMANTIC MEMORY (Temporal Cortex)
# ============================================================================


class SemanticMemory:
    """
    Symuluje semantic memory - długoterminowa wiedza o świecie.

    Biologicznie:
    - Przechowuje ogólną wiedzę (nie konkretne epizody)
    - "Trójkąt ma 3 wierzchołki"
    - "Kwadrat ma 4 wierzchołki"
    - Powolne uczenie się (konsolidacja)
    """

    def __init__(self):
        # Knowledge base (można to rozbudować!)
        self.knowledge = {
            "shapes": {
                3: {
                    "name": "triangle",
                    "properties": ["3 vertices", "3 edges", "polygon"],
                    "examples_seen": 0,
                    "confidence": 0.5,
                },
                4: {
                    "name": "square",
                    "properties": ["4 vertices", "4 edges", "polygon", "right angles"],
                    "examples_seen": 0,
                    "confidence": 0.5,
                },
                5: {
                    "name": "pentagon",
                    "properties": ["5 vertices", "5 edges", "polygon"],
                    "examples_seen": 0,
                    "confidence": 0.5,
                },
            }
        }

    def query(self, num_vertices):
        """Zapytaj: jaki kształt ma tyle wierzchołków?"""
        if num_vertices in self.knowledge["shapes"]:
            return self.knowledge["shapes"][num_vertices]
        else:
            return {"name": "unknown", "properties": [], "confidence": 0.0}

    def update(self, num_vertices, correct=True):
        """
        Aktualizuj wiedzę (uczenie się).
        Jeśli zobaczyłeś kształt i został potwierdzony.
        """
        if num_vertices in self.knowledge["shapes"]:
            self.knowledge["shapes"][num_vertices]["examples_seen"] += 1

            # Zwiększ confidence (powoli)
            current_conf = self.knowledge["shapes"][num_vertices]["confidence"]
            if correct:
                new_conf = current_conf + 0.1 * (
                    1.0 - current_conf
                )  # asymptotic to 1.0
            else:
                new_conf = current_conf * 0.9  # decay

            self.knowledge["shapes"][num_vertices]["confidence"] = new_conf

    def get_all_knowledge(self):
        """Zwróć całą wiedzę."""
        return self.knowledge


# ============================================================================
# MODULE 6: ATTENTION SYSTEM
# ============================================================================


class AttentionSystem(nn.Module):
    """
    Symuluje attention - co jest ważne, na co się skupić.

    Biologicznie:
    - Parietal + Frontal cortex
    - Top-down (goal-driven) i bottom-up (saliency-driven)
    - Moduluje processing w innych obszarach
    """

    def __init__(self):
        super().__init__()

        # Saliency map (co jest "interesujące")
        self.saliency_detector = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, corner_features):
        """
        Wygeneruj mapę uwagi.

        Input: corner_features [batch, 64, 128, 128]
        Output: attention_map [batch, 1, 128, 128]
        """
        attention_map = self.saliency_detector(corner_features)
        return attention_map


# ============================================================================
# MODULE 7: CENTRAL EXECUTIVE / GLOBAL WORKSPACE
# ============================================================================


class CentralExecutive:
    """
    Symuluje "central executive" / "global workspace".

    Biologicznie:
    - Prefrontal cortex + parietal
    - Integruje informacje z różnych modułów
    - Podejmuje decyzje
    - "Świadomość" emerguje tutaj (informacje w global workspace)
    """

    def __init__(self, working_memory, semantic_memory):
        self.working_memory = working_memory
        self.semantic_memory = semantic_memory
        self.decision_history = []

    def integrate_and_decide(self, perception_data):
        """
        Integruj dane z percepcji i podejmij decyzję.

        perception_data = {
            'numerosity_probs': tensor,
            'exact_count': int,
            'corner_heatmap': tensor,
            'attention_map': tensor
        }
        """
        # 1. Odczytaj exact count z Number Sense
        num_vertices = perception_data["exact_count"].item()

        # 2. Dodaj do working memory
        self.working_memory.update(
            {
                "type": "perception",
                "num_vertices": num_vertices,
                "timestamp": len(self.decision_history),
            },
            importance=1.0,
        )

        # 3. Query semantic memory
        shape_knowledge = self.semantic_memory.query(num_vertices)

        # 4. Dodaj wiedzę do working memory
        self.working_memory.update(
            {
                "type": "knowledge",
                "shape_name": shape_knowledge["name"],
                "confidence": shape_knowledge["confidence"],
            },
            importance=shape_knowledge["confidence"],
        )

        # 5. Podejmij decyzję
        decision = {
            "shape": shape_knowledge["name"],
            "num_vertices": num_vertices,
            "confidence": shape_knowledge["confidence"],
            "reasoning": f"Detected {num_vertices} vertices → {shape_knowledge['name']}",
        }

        # 6. Zapisz w historii
        self.decision_history.append(decision)

        return decision


# ============================================================================
# COMPLETE COGNITIVE ARCHITECTURE
# ============================================================================


class ShapeCognitiveArchitecture(nn.Module):
    """
    Kompletna cognitive architecture dla rozpoznawania kształtów.

    Symuluje:
    - Visual processing (V1 → V4)
    - Number sense (IPS)
    - Working memory (PFC)
    - Semantic memory (temporal)
    - Attention
    - Central executive
    """

    def __init__(self):
        super().__init__()

        # Neural modules
        self.v1_v2 = EarlyVisualCortex()
        self.v4_it = ShapeProcessor()
        self.ips = NumberSense()
        self.attention = AttentionSystem()

        # Symbolic/buffer modules (nie mają gradient!)
        self.working_memory = WorkingMemory(capacity=7)
        self.semantic_memory = SemanticMemory()
        self.central_executive = CentralExecutive(
            self.working_memory, self.semantic_memory
        )

    def forward(self, image, verbose=False):
        """
        Pełny forward pass przez całą cognitive architecture.

        Input: image [batch, 3, 512, 512]
        Output: decision dict
        """
        batch_size = image.shape[0]
        decisions = []

        for b in range(batch_size):
            img = image[b : b + 1]  # [1, 3, 512, 512]

            if verbose:
                print(f"\n{'='*60}")
                print(f"PROCESSING IMAGE {b}")
                print(f"{'='*60}")

            # 1. EARLY VISUAL CORTEX (V1/V2)
            if verbose:
                print("  [V1/V2] Detecting edges and orientations...")
            edge_features = self.v1_v2(img)

            # 2. SHAPE PROCESSING (V4/IT)
            if verbose:
                print("  [V4/IT] Detecting corners and contours...")
            corner_features, corner_heatmap, global_features = self.v4_it(edge_features)

            # 3. ATTENTION
            if verbose:
                print("  [Attention] Generating saliency map...")
            attention_map = self.attention(corner_features)

            # 4. NUMBER SENSE (IPS)
            if verbose:
                print("  [IPS] Estimating numerosity...")
            numerosity_probs, exact_count = self.ips(global_features, corner_heatmap)

            if verbose:
                print(f"    → Exact count: {exact_count.item()} vertices")
                print(
                    f"    → ANS distribution: {numerosity_probs[0, :5].detach().cpu().numpy()}"
                )

            # 5. CENTRAL EXECUTIVE - Integration & Decision
            if verbose:
                print("  [Central Executive] Integrating information...")

            perception_data = {
                "numerosity_probs": numerosity_probs,
                "exact_count": exact_count,
                "corner_heatmap": corner_heatmap,
                "attention_map": attention_map,
            }

            decision = self.central_executive.integrate_and_decide(perception_data)

            if verbose:
                print(f"  [Decision] {decision['reasoning']}")
                print(f"    → Shape: {decision['shape']}")
                print(f"    → Confidence: {decision['confidence']:.2f}")

            decisions.append(decision)

        return decisions


# ============================================================================
# TRAINING (yes, części cognitive architecture można trenować!)
# ============================================================================


def train_cognitive_architecture(model, dataloader, num_epochs=50, device="cpu"):
    """
    Trenuj neural części architecture.
    Symbolic części (working memory, semantic) aktualizują się automatycznie.
    """
    model.to(device)

    # Trenuj tylko neural modules
    neural_params = (
        list(model.v1_v2.parameters())
        + list(model.v4_it.parameters())
        + list(model.ips.parameters())
        + list(model.attention.parameters())
    )

    optimizer = torch.optim.Adam(neural_params, lr=0.001)

    print(f"\n{'='*70}")
    print("TRAINING COGNITIVE ARCHITECTURE")
    print(f"{'='*70}")

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0

        for images, _ in dataloader:
            images = images.to(device)

            # Forward przez neural modules
            edge_features = model.v1_v2(images)
            corner_features, corner_heatmap, global_features = model.v4_it(
                edge_features
            )
            numerosity_probs, exact_count = model.ips(global_features, corner_heatmap)

            # Loss: encourage corner detection
            # (to jest self-supervised - używamy własnej detekcji jako target)
            loss_corner = torch.mean((corner_heatmap - 0.5) ** 2)  # encourage peaks

            # Loss: encourage confident numerosity
            loss_numerosity = -torch.mean(
                torch.max(numerosity_probs, dim=1)[0]
            )  # encourage confidence

            total_loss_batch = loss_corner + 0.1 * loss_numerosity

            optimizer.zero_grad()
            total_loss_batch.backward()
            optimizer.step()

            total_loss += total_loss_batch.item()

        avg_loss = total_loss / len(dataloader)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs}: Loss = {avg_loss:.4f}")

    return model


# ============================================================================
# MAIN - DEMO
# ============================================================================

if __name__ == "__main__":
    from torchvision import transforms
    from analyze_both import ShapeDataset
    from torch.utils.data import DataLoader

    # Load data
    TRIANGLES_DIR = "triangles_dataset"
    SQUARES_DIR = "squares_dataset"

    transform = transforms.ToTensor()

    tri_dataset = ShapeDataset(TRIANGLES_DIR, transform=transform)
    sq_dataset = ShapeDataset(SQUARES_DIR, transform=transform)

    # Single samples for demo
    tri_sample = tri_dataset[0][0].unsqueeze(0)  # [1, 3, 512, 512]
    sq_sample = sq_dataset[0][0].unsqueeze(0)

    # Create architecture
    print("Tworzę Cognitive Architecture...")
    cogarch = ShapeCognitiveArchitecture()

    print(
        f"\nLiczba parametrów (neural modules): {sum(p.numel() for p in cogarch.parameters()):,}"
    )

    # Test na pojedynczych obrazach
    print("\n" + "=" * 70)
    print("TEST NA TRÓJKĄCIE")
    print("=" * 70)

    decision_tri = cogarch(tri_sample, verbose=True)

    print("\n" + "=" * 70)
    print("TEST NA KWADRACIE")
    print("=" * 70)

    decision_sq = cogarch(sq_sample, verbose=True)

    print("\n" + "=" * 70)
    print("GOTOWE!")
    print("=" * 70)
