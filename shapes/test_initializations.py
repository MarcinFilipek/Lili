# test_initializations.py

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ============================================================================
# 1. DATASET
# ============================================================================


class ShapeDataset(Dataset):
    """Dataset do wczytywania kształtów z folderu."""

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.image_files = [f for f in os.listdir(data_dir) if f.endswith(".png")]
        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, self.image_files[idx]


# ============================================================================
# 2. MODEL Z RÓŻNYMI INICJALIZACJAMI
# ============================================================================


class BetterConvNet(nn.Module):
    def __init__(self, init_method="kaiming"):
        super(BetterConvNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=11, stride=2, padding=5)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2, 2)

        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(512)

        self.relu = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Zapisz metodę dla późniejszego użycia
        self.init_method = init_method

        # Zastosuj inicjalizację
        self._initialize_weights(init_method)

    def _initialize_weights(self, method):
        """Inicjalizuj wagi wybraną metodą."""
        print(f"  Inicjalizuję wagi metodą: {method}")

        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if method == "kaiming":
                    init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

                elif method == "kaiming_uniform":
                    init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="relu")

                elif method == "xavier":
                    init.xavier_normal_(m.weight)

                elif method == "xavier_uniform":
                    init.xavier_uniform_(m.weight)

                elif method == "orthogonal":
                    init.orthogonal_(m.weight)

                elif method == "sparse":
                    with torch.no_grad():
                        # Inicjalizuj normalnie
                        init.normal_(m.weight, mean=0, std=0.01)
                        # Wyzeruj 90% losowo wybranych wag
                        mask = torch.rand_like(m.weight) > 0.9  # 10% True, 90% False
                        m.weight.data *= mask.float()

                elif method == "normal_small":
                    init.normal_(m.weight, mean=0, std=0.01)

                elif method == "normal_medium":
                    init.normal_(m.weight, mean=0, std=0.1)

                elif method == "normal_large":
                    init.normal_(m.weight, mean=0, std=0.5)

                elif method == "uniform_small":
                    init.uniform_(m.weight, -0.05, 0.05)

                elif method == "uniform_medium":
                    init.uniform_(m.weight, -0.1, 0.1)

                elif method == "uniform_large":
                    init.uniform_(m.weight, -0.5, 0.5)

                elif method == "gabor":
                    if "conv1" in name:  # Tylko pierwsza warstwa
                        self._init_gabor_like(m)
                    else:
                        init.kaiming_normal_(
                            m.weight, mode="fan_out", nonlinearity="relu"
                        )

                elif method == "edge_detector":
                    if "conv1" in name:  # Tylko pierwsza warstwa
                        self._init_edge_detectors(m)
                    else:
                        init.kaiming_normal_(
                            m.weight, mode="fan_out", nonlinearity="relu"
                        )

                else:
                    raise ValueError(f"Unknown init method: {method}")

                # Bias zawsze zero
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _init_gabor_like(self, conv_layer):
        """Inicjalizuj filtry jako Gabor-like (podobne do V1)."""
        n_filters = conv_layer.weight.shape[0]
        n_channels = conv_layer.weight.shape[1]
        kernel_size = conv_layer.weight.shape[2]

        # Generuj Gabor filtry z różnymi orientacjami
        n_orientations = min(8, n_filters)  # 8 orientacji
        orientations = np.linspace(0, np.pi, n_orientations, endpoint=False)

        # Różne częstotliwości
        frequencies = [2.0, 4.0, 6.0]

        filter_idx = 0
        for freq in frequencies:
            for theta in orientations:
                if filter_idx >= n_filters:
                    break

                # Parametry Gabor
                sigma = kernel_size / 4.0
                lambd = kernel_size / freq
                gamma = 0.5
                psi = 0

                # Generuj Gabor kernel
                gabor_kernel = self._create_gabor_kernel(
                    kernel_size, sigma, theta, lambd, gamma, psi
                )

                # Przypisz do wszystkich kanałów wejściowych
                for c in range(n_channels):
                    conv_layer.weight.data[filter_idx, c, :, :] = torch.from_numpy(
                        gabor_kernel
                    )

                filter_idx += 1

        # Jeśli zostały filtry, wypełnij losowo
        if filter_idx < n_filters:
            remaining_filters = conv_layer.weight[filter_idx:, :, :, :]
            init.kaiming_normal_(remaining_filters, mode="fan_out", nonlinearity="relu")

    def _create_gabor_kernel(self, size, sigma, theta, lambd, gamma, psi):
        """Tworzy Gabor kernel."""
        sigma_x = sigma
        sigma_y = sigma / gamma

        # Siatka współrzędnych
        center = size // 2
        x = np.arange(size) - center
        y = np.arange(size) - center
        X, Y = np.meshgrid(x, y)

        # Rotacja
        x_theta = X * np.cos(theta) + Y * np.sin(theta)
        y_theta = -X * np.sin(theta) + Y * np.cos(theta)

        # Gabor
        gb = np.exp(
            -0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)
        ) * np.cos(2 * np.pi * x_theta / lambd + psi)

        # Normalizuj
        gb = gb / (np.abs(gb).max() + 1e-6)

        return gb.astype(np.float32)

    def _init_edge_detectors(self, conv_layer):
        """Inicjalizuj jako klasyczne detektory krawędzi."""
        n_filters = conv_layer.weight.shape[0]
        n_channels = conv_layer.weight.shape[1]
        kernel_size = conv_layer.weight.shape[2]

        # Definiuj podstawowe filtry
        if kernel_size >= 3:
            # Sobel X
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)

            # Sobel Y
            sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

            # Laplacian
            laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

            # Diagonal edges
            diag1 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype=np.float32)
            diag2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=np.float32)

            # Pad do większego kernel_size jeśli potrzeba
            basic_filters = [sobel_x, sobel_y, laplacian, diag1, diag2]

            filter_idx = 0
            for basic_filter in basic_filters:
                if filter_idx >= n_filters:
                    break

                # Pad do odpowiedniego rozmiaru
                if kernel_size > 3:
                    pad_size = (kernel_size - 3) // 2
                    padded_filter = np.pad(basic_filter, pad_size, mode="constant")
                else:
                    padded_filter = basic_filter

                # Przypisz do wszystkich kanałów
                for c in range(n_channels):
                    conv_layer.weight.data[filter_idx, c, :, :] = torch.from_numpy(
                        padded_filter
                    )

                filter_idx += 1

            # Wypełnij pozostałe losowo
            if filter_idx < n_filters:
                remaining_filters = conv_layer.weight[filter_idx:, :, :, :]
                init.kaiming_normal_(
                    remaining_filters, mode="fan_out", nonlinearity="relu"
                )
        else:
            # Dla małych kerneli użyj kaiming
            init.kaiming_normal_(conv_layer.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)

        x = self.relu(self.bn5(self.conv5(x)))
        x = self.pool5(x)

        x = self.relu(self.bn6(self.conv6(x)))

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        return x


# ============================================================================
# 3. EKSTRAKCJA AKTYWACJI
# ============================================================================


class ActivationExtractor:
    """Klasa do wyciągania aktywacji z wybranych warstw."""

    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []

    def register_hooks(self, layer_names):
        for name in layer_names:
            layer = getattr(self.model, name)
            hook = layer.register_forward_hook(self._get_activation(name))
            self.hooks.append(hook)

    def _get_activation(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach().cpu()

        return hook

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


# ============================================================================
# 4. ANALIZA POJEDYNCZEGO MODELU
# ============================================================================


def analyze_model_on_shapes(model, triangles_dir, squares_dir, batch_size=16):
    """Analizuje dany model na obu datasetach."""

    transform = transforms.Compose([transforms.ToTensor()])

    tri_dataset = ShapeDataset(triangles_dir, transform=transform)
    sq_dataset = ShapeDataset(squares_dir, transform=transform)

    tri_loader = DataLoader(
        tri_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    sq_loader = DataLoader(
        sq_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    model.eval()

    extractor = ActivationExtractor(model)
    extractor.register_hooks(["conv1", "conv2", "conv3", "conv4", "conv5", "conv6"])

    # Zbierz aktywacje dla trójkątów
    tri_activations = {
        "conv1": [],
        "conv2": [],
        "conv3": [],
        "conv4": [],
        "conv5": [],
        "conv6": [],
        "output": [],
    }

    with torch.no_grad():
        for images, _ in tqdm(tri_loader, desc="  Trójkąty", leave=False):
            output = model(images)

            for layer_name in ["conv1", "conv2", "conv3", "conv4", "conv5", "conv6"]:
                act = extractor.activations[layer_name]
                act_pooled = act.mean(dim=[2, 3])
                tri_activations[layer_name].append(act_pooled.numpy())

            tri_activations["output"].append(output.numpy())

    # Zbierz aktywacje dla kwadratów
    sq_activations = {
        "conv1": [],
        "conv2": [],
        "conv3": [],
        "conv4": [],
        "conv5": [],
        "conv6": [],
        "output": [],
    }

    with torch.no_grad():
        for images, _ in tqdm(sq_loader, desc="  Kwadraty", leave=False):
            output = model(images)

            for layer_name in ["conv1", "conv2", "conv3", "conv4", "conv5", "conv6"]:
                act = extractor.activations[layer_name]
                act_pooled = act.mean(dim=[2, 3])
                sq_activations[layer_name].append(act_pooled.numpy())

            sq_activations["output"].append(output.numpy())

    # Połącz batche
    for key in tri_activations:
        tri_activations[key] = np.vstack(tri_activations[key])
        sq_activations[key] = np.vstack(sq_activations[key])

    extractor.remove_hooks()

    return tri_activations, sq_activations


# ============================================================================
# 5. OBLICZANIE METRYK
# ============================================================================


def compute_metrics(tri_act, sq_act, layer="output"):
    """Oblicza metryki separacji dla danej warstwy."""
    tri = tri_act[layer]
    sq = sq_act[layer]

    tri_mean = tri.mean(axis=0)
    sq_mean = sq.mean(axis=0)
    tri_std = tri.std(axis=0)
    sq_std = sq.std(axis=0)

    # Korelacja
    correlation = np.corrcoef(tri_mean, sq_mean)[0, 1]

    # Selektywność
    selectivity = np.abs(tri_mean - sq_mean)

    # T-test
    from scipy.stats import ttest_ind

    p_values = []
    for i in range(tri.shape[1]):
        _, p_val = ttest_ind(tri[:, i], sq[:, i])
        p_values.append(p_val)
    p_values = np.array(p_values)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((tri_std**2 + sq_std**2) / 2)
    cohens_d = (tri_mean - sq_mean) / (pooled_std + 1e-6)

    metrics = {
        "correlation": correlation,
        "mean_selectivity": selectivity.mean(),
        "max_selectivity": selectivity.max(),
        "significant_neurons": (p_values < 0.05).sum(),
        "total_neurons": len(tri_mean),
        "large_effect_size": (np.abs(cohens_d) > 0.8).sum(),
        "very_large_effect_size": (np.abs(cohens_d) > 1.2).sum(),
    }

    return metrics


# ============================================================================
# 6. GŁÓWNY EKSPERYMENT
# ============================================================================


def run_initialization_experiment(triangles_dir, squares_dir, seed=42):
    """Testuje wszystkie metody inicjalizacji."""

    # Lista metod do przetestowania
    init_methods = [
        "kaiming",
        "kaiming_uniform",
        "xavier",
        "xavier_uniform",
        "orthogonal",
        "sparse",
        "normal_small",
        "normal_medium",
        "normal_large",
        "uniform_small",
        "uniform_medium",
        "uniform_large",
        "gabor",
        "edge_detector",
    ]

    results = {}

    print("=" * 70)
    print("TESTOWANIE RÓŻNYCH INICJALIZACJI")
    print("=" * 70)
    print(f"Seed: {seed}")
    print(f"Liczba metod: {len(init_methods)}\n")

    for method in init_methods:
        print(f"\n{'='*70}")
        print(f"Metoda: {method.upper()}")
        print(f"{'='*70}")

        # Ustaw seed dla reprodukowalności
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Stwórz model
        model = BetterConvNet(init_method=method)
        model.eval()

        # Analizuj
        print("Analizuję aktywacje...")
        tri_act, sq_act = analyze_model_on_shapes(
            model, triangles_dir, squares_dir, batch_size=16
        )

        # Oblicz metryki dla wszystkich warstw
        method_results = {}
        layers = ["conv1", "conv2", "conv3", "conv4", "conv5", "conv6", "output"]

        for layer in layers:
            metrics = compute_metrics(tri_act, sq_act, layer)
            method_results[layer] = metrics

            # Drukuj tylko dla output
            if layer == "output":
                print(f"\nWyniki dla {layer}:")
                print(f"  Korelacja: {metrics['correlation']:.4f}")
                print(f"  Średnia selektywność: {metrics['mean_selectivity']:.4f}")
                print(
                    f"  Neurony znaczące: {metrics['significant_neurons']}/{metrics['total_neurons']} "
                    f"({metrics['significant_neurons']/metrics['total_neurons']*100:.1f}%)"
                )
                print(f"  Duży effect size (|d|>0.8): {metrics['large_effect_size']}")
                print(
                    f"  Bardzo duży effect size (|d|>1.2): {metrics['very_large_effect_size']}"
                )

        results[method] = method_results

        # Zapisz checkpointy dla najciekawszych metod
        if method in ["gabor", "edge_detector", "orthogonal"]:
            torch.save(model.state_dict(), f"init_{method}_seed{seed}.pth")

    return results


# ============================================================================
# 7. WIZUALIZACJE
# ============================================================================


def visualize_results(results, save_prefix="init_comparison_"):
    """Wizualizuj wyniki wszystkich inicjalizacji."""

    methods = list(results.keys())
    layers = ["conv1", "conv2", "conv3", "conv4", "conv5", "conv6", "output"]

    # ========================================================================
    # Panel 1: Korelacje dla wszystkich warstw
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Subplot 1: Korelacje w output layer
    ax = axes[0, 0]
    correlations = [results[m]["output"]["correlation"] for m in methods]
    colors = [
        "red" if c < 0.5 else "orange" if c < 0.8 else "gray" for c in correlations
    ]

    y_pos = np.arange(len(methods))
    ax.barh(y_pos, correlations, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel("Korelacja", fontsize=11)
    ax.set_title("Korelacja w output layer", fontsize=12, fontweight="bold")
    ax.axvline(0.5, color="red", linestyle="--", alpha=0.5, label="r=0.5")
    ax.axvline(0.8, color="orange", linestyle="--", alpha=0.5, label="r=0.8")
    ax.legend()
    ax.grid(True, alpha=0.2, axis="x")

    # Subplot 2: % neuronów znaczących
    ax = axes[0, 1]
    significant_pct = [
        results[m]["output"]["significant_neurons"]
        / results[m]["output"]["total_neurons"]
        * 100
        for m in methods
    ]
    colors = [
        "green" if p > 80 else "yellow" if p > 50 else "gray" for p in significant_pct
    ]

    ax.barh(y_pos, significant_pct, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel("% neuronów znaczących (p<0.05)", fontsize=11)
    ax.set_title("Neurony statystycznie znaczące", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="x")

    # Subplot 3: Effect sizes
    ax = axes[1, 0]
    large_effect = [results[m]["output"]["large_effect_size"] for m in methods]
    very_large_effect = [
        results[m]["output"]["very_large_effect_size"] for m in methods
    ]

    x = np.arange(len(methods))
    width = 0.35
    ax.bar(
        x - width / 2,
        large_effect,
        width,
        label="|d| > 0.8",
        alpha=0.7,
        color="steelblue",
    )
    ax.bar(
        x + width / 2,
        very_large_effect,
        width,
        label="|d| > 1.2",
        alpha=0.7,
        color="darkblue",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Liczba neuronów", fontsize=11)
    ax.set_title("Effect sizes (Cohen's d)", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")

    # Subplot 4: Heatmap korelacji across layers
    ax = axes[1, 1]

    corr_matrix = np.zeros((len(methods), len(layers)))
    for i, method in enumerate(methods):
        for j, layer in enumerate(layers):
            corr_matrix[i, j] = results[method][layer]["correlation"]

    im = ax.imshow(corr_matrix, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=8)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, fontsize=8)
    ax.set_title("Korelacje across all layers", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Korelacja")

    # Zaznacz najniższe wartości
    for i in range(len(methods)):
        for j in range(len(layers)):
            text = ax.text(
                j,
                i,
                f"{corr_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if corr_matrix[i, j] < 0.5 else "black",
                fontsize=6,
            )

    plt.tight_layout()
    plt.savefig(f"{save_prefix}overview.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ========================================================================
    # Panel 2: Ranking table
    # ========================================================================
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")

    # Stwórz tabelę z rankingiem
    table_data = []
    for method in methods:
        row = [
            method,
            f"{results[method]['output']['correlation']:.3f}",
            f"{results[method]['output']['mean_selectivity']:.4f}",
            f"{results[method]['output']['significant_neurons']}/{results[method]['output']['total_neurons']}",
            f"{results[method]['output']['large_effect_size']}",
            f"{results[method]['output']['very_large_effect_size']}",
        ]
        table_data.append(row)

    # Sortuj po korelacji (rosnąco - niższa lepsza)
    table_data.sort(key=lambda x: float(x[1]))

    # Dodaj ranking
    for i, row in enumerate(table_data, 1):
        row.insert(0, str(i))

    headers = [
        "Rank",
        "Method",
        "Correlation",
        "Mean Sel.",
        "Significant",
        "|d|>0.8",
        "|d|>1.2",
    ]

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        colWidths=[0.08, 0.20, 0.12, 0.12, 0.15, 0.10, 0.10],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Koloruj header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Koloruj top 3
    for i in range(1, min(4, len(table_data) + 1)):
        for j in range(len(headers)):
            if j == 0:  # Rank column
                table[(i, j)].set_facecolor(
                    "#FFD700" if i == 1 else "#C0C0C0" if i == 2 else "#CD7F32"
                )

    plt.title(
        "Ranking metod inicjalizacji (output layer)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.savefig(f"{save_prefix}ranking.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\nZapisano wizualizacje:")
    print(f"  - {save_prefix}overview.png")
    print(f"  - {save_prefix}ranking.png")


def visualize_filters(model, init_method, save_prefix="filters_"):
    """Wizualizuj filtry pierwszej warstwy."""
    filters = model.conv1.weight.data[:32, 0, :, :].cpu().numpy()

    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(filters[i], cmap="gray", vmin=filters.min(), vmax=filters.max())
        ax.axis("off")

    plt.suptitle(f"Conv1 filters - {init_method}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}{init_method}.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
# 8. MAIN
# ============================================================================

if __name__ == "__main__":
    # Parametry
    TRIANGLES_DIR = "triangles_dataset"
    SQUARES_DIR = "squares_dataset"
    SEED = 42

    # Sprawdź czy foldery istnieją
    if not os.path.exists(TRIANGLES_DIR):
        print(f"BŁĄD: Folder {TRIANGLES_DIR} nie istnieje!")
        exit()

    if not os.path.exists(SQUARES_DIR):
        print(f"BŁĄD: Folder {SQUARES_DIR} nie istnieje!")
        exit()

    # Uruchom eksperyment
    results = run_initialization_experiment(TRIANGLES_DIR, SQUARES_DIR, seed=SEED)

    # Wizualizuj
    print("\n" + "=" * 70)
    print("TWORZĘ WIZUALIZACJE")
    print("=" * 70)
    visualize_results(results, save_prefix="init_comparison_")

    # Wizualizuj filtry dla wybranych metod
    print("\nWizualizuję filtry dla wybranych metod...")
    interesting_methods = ["gabor", "edge_detector", "kaiming", "orthogonal"]

    for method in interesting_methods:
        if method in results:
            torch.manual_seed(SEED)
            np.random.seed(SEED)
            model = BetterConvNet(init_method=method)
            visualize_filters(model, method, save_prefix="filters_")
            print(f"  Zapisano: filters_{method}.png")

    # Podsumowanie finalne
    print("\n" + "=" * 70)
    print("FINALNE PODSUMOWANIE")
    print("=" * 70)

    # Znajdź najlepszą metodę (najniższa korelacja)
    best_method = min(results.keys(), key=lambda m: results[m]["output"]["correlation"])
    worst_method = max(
        results.keys(), key=lambda m: results[m]["output"]["correlation"]
    )

    print(f"\n🏆 NAJLEPSZA METODA: {best_method}")
    print(f"   Korelacja: {results[best_method]['output']['correlation']:.4f}")
    print(
        f"   Znaczące neurony: {results[best_method]['output']['significant_neurons']}/{results[best_method]['output']['total_neurons']}"
    )
    print(f"   Duży effect size: {results[best_method]['output']['large_effect_size']}")

    print(f"\n❌ NAJGORSZA METODA: {worst_method}")
    print(f"   Korelacja: {results[worst_method]['output']['correlation']:.4f}")
    print(
        f"   Znaczące neurony: {results[worst_method]['output']['significant_neurons']}/{results[worst_method]['output']['total_neurons']}"
    )
    print(
        f"   Duży effect size: {results[worst_method]['output']['large_effect_size']}"
    )

    print("\n" + "=" * 70)
    print("GOTOWE!")
    print("=" * 70)
