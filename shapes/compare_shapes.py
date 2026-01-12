# compare_shapes.py

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

# ============================================================================
# 1. FUNKCJE DO ŁADOWANIA DANYCH
# ============================================================================


def save_activations(activations, filenames, output_file):
    """
    Zapisz aktywacje do pliku pickle.
    """
    data = {"activations": activations, "filenames": filenames}
    with open(output_file, "wb") as f:
        pickle.dump(data, f)
    print(f"Zapisano aktywacje do: {output_file}")


def load_activations(input_file):
    """
    Wczytaj aktywacje z pliku pickle.
    """
    with open(input_file, "rb") as f:
        data = pickle.load(f)
    print(f"Wczytano aktywacje z: {input_file}")
    return data["activations"], data["filenames"]


# ============================================================================
# 2. FUNKCJE STATYSTYCZNE
# ============================================================================


def compute_selectivity_metrics(tri_act, sq_act, layer="conv3"):
    """
    Oblicza metryki selektywności dla danej warstwy.

    Returns:
        dict z metrykami
    """
    tri = tri_act[layer]  # shape: [n_images, n_neurons]
    sq = sq_act[layer]

    # Średnie i std
    tri_mean = tri.mean(axis=0)
    sq_mean = sq.mean(axis=0)
    tri_std = tri.std(axis=0)
    sq_std = sq.std(axis=0)

    # Selektywność = różnica w średnich
    selectivity = np.abs(tri_mean - sq_mean)

    # Niezmienniczość = odwrotność std (niskie std = wysoka niezmienniczość)
    tri_invariance = 1.0 / (tri_std + 1e-6)
    sq_invariance = 1.0 / (sq_std + 1e-6)

    # T-test dla każdego neuronu (czy różnice są znaczące?)
    p_values = []
    t_statistics = []
    for i in range(tri.shape[1]):
        t_stat, p_val = ttest_ind(tri[:, i], sq[:, i])
        t_statistics.append(t_stat)
        p_values.append(p_val)

    p_values = np.array(p_values)
    t_statistics = np.array(t_statistics)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((tri_std**2 + sq_std**2) / 2)
    cohens_d = (tri_mean - sq_mean) / (pooled_std + 1e-6)

    # Korelacja między średnimi aktywacjami
    correlation = np.corrcoef(tri_mean, sq_mean)[0, 1]

    # Separowalność (jaka część neuronów jest "znacząco różna"?)
    significant_neurons = (p_values < 0.05).sum()

    metrics = {
        "tri_mean": tri_mean,
        "sq_mean": sq_mean,
        "tri_std": tri_std,
        "sq_std": sq_std,
        "selectivity": selectivity,
        "tri_invariance": tri_invariance,
        "sq_invariance": sq_invariance,
        "p_values": p_values,
        "t_statistics": t_statistics,
        "cohens_d": cohens_d,
        "correlation": correlation,
        "significant_neurons": significant_neurons,
        "total_neurons": len(tri_mean),
    }

    return metrics


def print_statistics(metrics, layer_name):
    """
    Wydrukuj statystyki w czytelnej formie.
    """
    print(f"\n{'='*70}")
    print(f"STATYSTYKI DLA WARSTWY: {layer_name}")
    print(f"{'='*70}")

    print(f"\nLiczba neuronów: {metrics['total_neurons']}")

    print(f"\n--- PODSTAWOWE STATYSTYKI ---")
    print(f"Trójkąty:")
    print(
        f"  Średnia aktywacja: {metrics['tri_mean'].mean():.4f} ± {metrics['tri_mean'].std():.4f}"
    )
    print(f"  Średnie std: {metrics['tri_std'].mean():.4f}")
    print(
        f"  Min/Max: {metrics['tri_mean'].min():.4f} / {metrics['tri_mean'].max():.4f}"
    )

    print(f"\nKwadraty:")
    print(
        f"  Średnia aktywacja: {metrics['sq_mean'].mean():.4f} ± {metrics['sq_mean'].std():.4f}"
    )
    print(f"  Średnie std: {metrics['sq_std'].mean():.4f}")
    print(f"  Min/Max: {metrics['sq_mean'].min():.4f} / {metrics['sq_mean'].max():.4f}")

    print(f"\n--- SELEKTYWNOŚĆ ---")
    print(f"Średnia selektywność: {metrics['selectivity'].mean():.4f}")
    print(f"Maksymalna selektywność: {metrics['selectivity'].max():.4f}")
    print(f"Top 10% selektywności: {np.percentile(metrics['selectivity'], 90):.4f}")
    print(f"Top 5% selektywności: {np.percentile(metrics['selectivity'], 95):.4f}")

    print(f"\n--- ZNACZENIE STATYSTYCZNE ---")
    print(
        f"Neurony znacząco różne (p < 0.05): {metrics['significant_neurons']} / {metrics['total_neurons']} ({metrics['significant_neurons']/metrics['total_neurons']*100:.1f}%)"
    )
    print(
        f"Neurony wysoko znaczące (p < 0.01): {(metrics['p_values'] < 0.01).sum()} ({(metrics['p_values'] < 0.01).sum()/metrics['total_neurons']*100:.1f}%)"
    )
    print(
        f"Neurony bardzo wysoko znaczące (p < 0.001): {(metrics['p_values'] < 0.001).sum()} ({(metrics['p_values'] < 0.001).sum()/metrics['total_neurons']*100:.1f}%)"
    )

    print(f"\n--- EFFECT SIZE (Cohen's d) ---")
    print(f"Średni effect size: {np.abs(metrics['cohens_d']).mean():.4f}")
    print(
        f"Duży effect size (|d| > 0.8): {(np.abs(metrics['cohens_d']) > 0.8).sum()} neuronów"
    )
    print(
        f"Bardzo duży effect size (|d| > 1.2): {(np.abs(metrics['cohens_d']) > 1.2).sum()} neuronów"
    )

    print(f"\n--- KORELACJA ---")
    print(f"Korelacja Tri vs Sq: {metrics['correlation']:.4f}")
    if metrics["correlation"] > 0.9:
        print("  → Bardzo wysoka (neurony reagują podobnie)")
    elif metrics["correlation"] > 0.7:
        print("  → Wysoka (znaczące podobieństwo)")
    elif metrics["correlation"] > 0.5:
        print("  → Umiarkowana (częściowa separacja)")
    else:
        print("  → Niska (silna separacja kształtów!)")

    print(f"\n--- TOP 10 NAJBARDZIEJ SELEKTYWNYCH NEURONÓW ---")
    top_indices = metrics["selectivity"].argsort()[-10:][::-1]
    for rank, idx in enumerate(top_indices, 1):
        print(f"{rank:2d}. Neuron {idx:3d}:")
        print(
            f"    Tri: {metrics['tri_mean'][idx]:.4f} ± {metrics['tri_std'][idx]:.4f}"
        )
        print(f"    Sq:  {metrics['sq_mean'][idx]:.4f} ± {metrics['sq_std'][idx]:.4f}")
        print(
            f"    Sel: {metrics['selectivity'][idx]:.4f}, Cohen's d: {metrics['cohens_d'][idx]:.4f}, p: {metrics['p_values'][idx]:.6f}"
        )


# ============================================================================
# 3. WIZUALIZACJE ZAAWANSOWANE
# ============================================================================


def plot_comprehensive_comparison(tri_act, sq_act, layer="conv3", save_prefix=""):
    """
    Kompleksowa wizualizacja porównawcza.
    """
    metrics = compute_selectivity_metrics(tri_act, sq_act, layer)

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # ========================================================================
    # PANEL 1: Scatter plot - Mean activations
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(metrics["tri_mean"], metrics["sq_mean"], alpha=0.4, s=30, c="gray")

    # Zaznacz znaczące neurony
    significant = metrics["p_values"] < 0.05
    ax1.scatter(
        metrics["tri_mean"][significant],
        metrics["sq_mean"][significant],
        alpha=0.6,
        s=50,
        c="red",
        label="p < 0.05",
    )

    # Linia równej aktywacji
    max_val = max(metrics["tri_mean"].max(), metrics["sq_mean"].max())
    ax1.plot([0, max_val], [0, max_val], "k--", alpha=0.3, linewidth=2)

    ax1.set_xlabel("Średnia aktywacja: Trójkąty", fontsize=11)
    ax1.set_ylabel("Średnia aktywacja: Kwadraty", fontsize=11)
    ax1.set_title(
        f'{layer}: Porównanie aktywacji\n(r = {metrics["correlation"]:.3f})',
        fontsize=12,
        fontweight="bold",
    )
    ax1.legend()
    ax1.grid(True, alpha=0.2)

    # ========================================================================
    # PANEL 2: Histogram selektywności z p-values
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 1])

    # Dwa histogramy: znaczące vs nie-znaczące
    significant_sel = metrics["selectivity"][metrics["p_values"] < 0.05]
    not_significant_sel = metrics["selectivity"][metrics["p_values"] >= 0.05]

    ax2.hist(
        not_significant_sel,
        bins=30,
        alpha=0.5,
        label="p ≥ 0.05",
        color="gray",
        edgecolor="black",
    )
    ax2.hist(
        significant_sel,
        bins=30,
        alpha=0.7,
        label="p < 0.05",
        color="red",
        edgecolor="black",
    )

    ax2.axvline(
        metrics["selectivity"].mean(),
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f'Średnia: {metrics["selectivity"].mean():.3f}',
    )
    ax2.set_xlabel("Selektywność |μ_tri - μ_sq|", fontsize=11)
    ax2.set_ylabel("Liczba neuronów", fontsize=11)
    ax2.set_title("Rozkład selektywności", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.2, axis="y")

    # ========================================================================
    # PANEL 3: Cohen's d distribution
    # ========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(metrics["cohens_d"], bins=40, alpha=0.7, color="purple", edgecolor="black")
    ax3.axvline(0, color="black", linestyle="-", linewidth=1)
    ax3.axvline(0.8, color="green", linestyle="--", alpha=0.5, label="|d| = 0.8 (duży)")
    ax3.axvline(-0.8, color="green", linestyle="--", alpha=0.5)
    ax3.axvline(
        1.2, color="red", linestyle="--", alpha=0.5, label="|d| = 1.2 (bardzo duży)"
    )
    ax3.axvline(-1.2, color="red", linestyle="--", alpha=0.5)
    ax3.set_xlabel("Cohen's d (effect size)", fontsize=11)
    ax3.set_ylabel("Liczba neuronów", fontsize=11)
    ax3.set_title("Effect Size Distribution", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.2, axis="y")

    # ========================================================================
    # PANEL 4: P-value distribution (volcano plot style)
    # ========================================================================
    ax4 = fig.add_subplot(gs[0, 3])

    # -log10(p) vs effect size
    neg_log_p = -np.log10(metrics["p_values"] + 1e-300)  # avoid log(0)

    colors = ["red" if p < 0.05 else "gray" for p in metrics["p_values"]]
    ax4.scatter(metrics["cohens_d"], neg_log_p, alpha=0.5, s=30, c=colors)

    ax4.axhline(
        -np.log10(0.05), color="red", linestyle="--", alpha=0.5, label="p = 0.05"
    )
    ax4.axvline(0, color="black", linestyle="-", linewidth=1)

    ax4.set_xlabel("Cohen's d", fontsize=11)
    ax4.set_ylabel("-log₁₀(p-value)", fontsize=11)
    ax4.set_title("Volcano Plot", fontsize=12, fontweight="bold")
    ax4.legend()
    ax4.grid(True, alpha=0.2)

    # ========================================================================
    # PANEL 5: Top 20 neurons bar plot
    # ========================================================================
    ax5 = fig.add_subplot(gs[1, :2])

    top_20_idx = metrics["selectivity"].argsort()[-20:]
    x = np.arange(20)
    width = 0.35

    ax5.bar(
        x - width / 2,
        metrics["tri_mean"][top_20_idx],
        width,
        label="Trójkąty",
        alpha=0.8,
        color="blue",
    )
    ax5.bar(
        x + width / 2,
        metrics["sq_mean"][top_20_idx],
        width,
        label="Kwadraty",
        alpha=0.8,
        color="orange",
    )

    ax5.set_xlabel("Neuron (top 20 selektywnych)", fontsize=11)
    ax5.set_ylabel("Średnia aktywacja", fontsize=11)
    ax5.set_title(
        "Top 20 najbardziej selektywnych neuronów", fontsize=12, fontweight="bold"
    )
    ax5.set_xticks(x)
    ax5.set_xticklabels(
        [f"{idx}" for idx in top_20_idx], rotation=45, ha="right", fontsize=8
    )
    ax5.legend()
    ax5.grid(True, alpha=0.2, axis="y")

    # ========================================================================
    # PANEL 6: Selectivity vs Invariance scatter
    # ========================================================================
    ax6 = fig.add_subplot(gs[1, 2])

    scatter = ax6.scatter(
        metrics["selectivity"],
        metrics["tri_invariance"],
        c=metrics["tri_mean"],
        cmap="viridis",
        alpha=0.6,
        s=50,
    )

    # Zaznacz "idealne" neurony
    high_sel = np.percentile(metrics["selectivity"], 90)
    high_inv = np.percentile(metrics["tri_invariance"], 90)
    ideal = (metrics["selectivity"] > high_sel) & (metrics["tri_invariance"] > high_inv)

    ax6.scatter(
        metrics["selectivity"][ideal],
        metrics["tri_invariance"][ideal],
        color="red",
        s=200,
        marker="*",
        edgecolor="black",
        linewidth=1.5,
        label=f"Idealne ({ideal.sum()})",
        zorder=5,
    )

    ax6.set_xlabel("Selektywność", fontsize=11)
    ax6.set_ylabel("Niezmienniczość (trójkąty)", fontsize=11)
    ax6.set_title("Selektywność vs Niezmienniczość", fontsize=12, fontweight="bold")
    ax6.legend()
    ax6.grid(True, alpha=0.2)
    plt.colorbar(scatter, ax=ax6, label="μ_tri")

    # ========================================================================
    # PANEL 7: Variance comparison
    # ========================================================================
    ax7 = fig.add_subplot(gs[1, 3])

    ax7.scatter(metrics["tri_std"], metrics["sq_std"], alpha=0.5, s=30, c="gray")

    max_std = max(metrics["tri_std"].max(), metrics["sq_std"].max())
    ax7.plot([0, max_std], [0, max_std], "r--", alpha=0.5)

    ax7.set_xlabel("Std (Trójkąty)", fontsize=11)
    ax7.set_ylabel("Std (Kwadraty)", fontsize=11)
    ax7.set_title("Porównanie wariancji", fontsize=12, fontweight="bold")
    ax7.grid(True, alpha=0.2)

    # ========================================================================
    # PANEL 8: Heatmap top neurons
    # ========================================================================
    ax8 = fig.add_subplot(gs[2, :2])

    top_15_idx = metrics["selectivity"].argsort()[-15:]
    heatmap_data = np.vstack(
        [metrics["tri_mean"][top_15_idx], metrics["sq_mean"][top_15_idx]]
    )

    im = ax8.imshow(heatmap_data, aspect="auto", cmap="YlOrRd")
    ax8.set_yticks([0, 1])
    ax8.set_yticklabels(["Trójkąty", "Kwadraty"], fontsize=11)
    ax8.set_xticks(range(15))
    ax8.set_xticklabels(
        [f"N{idx}" for idx in top_15_idx], rotation=45, ha="right", fontsize=9
    )
    ax8.set_title("Heatmapa: Top 15 neuronów", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax8, label="Aktywacja")

    # ========================================================================
    # PANEL 9: Statistics text summary
    # ========================================================================
    ax9 = fig.add_subplot(gs[2, 2:])
    ax9.axis("off")

    summary_text = f"""
PODSUMOWANIE STATYSTYCZNE - {layer}
{'='*50}

Całkowita liczba neuronów: {metrics['total_neurons']}

AKTYWACJE:
  Trójkąty: μ = {metrics['tri_mean'].mean():.4f} ± {metrics['tri_mean'].std():.4f}
  Kwadraty: μ = {metrics['sq_mean'].mean():.4f} ± {metrics['sq_mean'].std():.4f}

SELEKTYWNOŚĆ:
  Średnia: {metrics['selectivity'].mean():.4f}
  Maksymalna: {metrics['selectivity'].max():.4f}
  Top 10%: {np.percentile(metrics['selectivity'], 90):.4f}

ZNACZENIE STATYSTYCZNE:
  p < 0.05:  {metrics['significant_neurons']:3d} ({metrics['significant_neurons']/metrics['total_neurons']*100:5.1f}%)
  p < 0.01:  {(metrics['p_values'] < 0.01).sum():3d} ({(metrics['p_values'] < 0.01).sum()/metrics['total_neurons']*100:5.1f}%)
  p < 0.001: {(metrics['p_values'] < 0.001).sum():3d} ({(metrics['p_values'] < 0.001).sum()/metrics['total_neurons']*100:5.1f}%)

EFFECT SIZE (Cohen's d):
  Średni |d|: {np.abs(metrics['cohens_d']).mean():.4f}
  Duży (|d| > 0.8): {(np.abs(metrics['cohens_d']) > 0.8).sum():3d} neuronów
  Bardzo duży (|d| > 1.2): {(np.abs(metrics['cohens_d']) > 1.2).sum():3d} neuronów

KORELACJA:
  r(Tri, Sq) = {metrics['correlation']:.4f}

INTERPRETACJA:
"""

    if metrics["correlation"] > 0.9:
        summary_text += "  ⚠ Bardzo wysoka korelacja - neurony nie różnicują\n  kształtów (prawdopodobnie sieć losowa)"
    elif metrics["correlation"] > 0.7:
        summary_text += "  → Wysoka korelacja - częściowa separacja"
    elif metrics["correlation"] > 0.5:
        summary_text += "  ✓ Umiarkowana separacja kształtów"
    else:
        summary_text += "  ✓✓ SILNA SEPARACJA! Neurony wyspecjalizowane!"

    if metrics["significant_neurons"] / metrics["total_neurons"] > 0.5:
        summary_text += "\n  ✓ Ponad połowa neuronów znacząco różnicuje"

    if (np.abs(metrics["cohens_d"]) > 0.8).sum() > 10:
        summary_text += f"\n  ✓ {(np.abs(metrics['cohens_d']) > 0.8).sum()} neuronów z dużym effect size"

    ax9.text(
        0.05,
        0.95,
        summary_text,
        transform=ax9.transAxes,
        fontsize=10,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    # ========================================================================
    # Zapisz
    # ========================================================================
    filename = f"{save_prefix}comprehensive_comparison_{layer}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nZapisano: {filename}")

    return metrics


def plot_all_layers_summary(tri_act, sq_act, save_prefix=""):
    """
    Podsumowanie wszystkich warstw na jednym wykresie.
    """
    layers = ["conv1", "conv2", "conv3", "output"]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    all_metrics = {}

    for idx, layer in enumerate(layers):
        metrics = compute_selectivity_metrics(tri_act, sq_act, layer)
        all_metrics[layer] = metrics

        # Górny rząd: Scatter plots
        ax = axes[0, idx]
        ax.scatter(metrics["tri_mean"], metrics["sq_mean"], alpha=0.4, s=20, c="gray")

        significant = metrics["p_values"] < 0.05
        ax.scatter(
            metrics["tri_mean"][significant],
            metrics["sq_mean"][significant],
            alpha=0.6,
            s=30,
            c="red",
        )

        max_val = max(metrics["tri_mean"].max(), metrics["sq_mean"].max())
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3)

        ax.set_xlabel("Trójkąty", fontsize=10)
        ax.set_ylabel("Kwadraty", fontsize=10)
        ax.set_title(
            f'{layer}\nr={metrics["correlation"]:.3f}, p<0.05: {metrics["significant_neurons"]}',
            fontsize=11,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.2)

        # Dolny rząd: Histogramy selektywności
        ax = axes[1, idx]
        ax.hist(
            metrics["selectivity"],
            bins=30,
            alpha=0.7,
            color="steelblue",
            edgecolor="black",
        )
        ax.axvline(
            metrics["selectivity"].mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f'μ={metrics["selectivity"].mean():.3f}',
        )
        ax.set_xlabel("Selektywność", fontsize=10)
        ax.set_ylabel("Liczba neuronów", fontsize=10)
        ax.set_title(f"Rozkład selektywności", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    filename = f"{save_prefix}all_layers_summary.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Zapisano: {filename}")

    return all_metrics


def plot_pca_comparison(tri_act, sq_act, layer="conv3", save_prefix=""):
    """
    PCA - wizualizacja przestrzeni aktywacji.
    """
    tri = tri_act[layer]
    sq = sq_act[layer]

    # Połącz dane
    all_data = np.vstack([tri, sq])
    labels = np.array(["Triangle"] * len(tri) + ["Square"] * len(sq))

    # PCA
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(all_data)

    # Wykres
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: PCA scatter
    ax = axes[0]
    for shape, color in [("Triangle", "blue"), ("Square", "orange")]:
        mask = labels == shape
        ax.scatter(
            transformed[mask, 0],
            transformed[mask, 1],
            alpha=0.5,
            s=30,
            c=color,
            label=shape,
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
    ax.set_title(f"{layer}: PCA projection", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.2)

    # Panel 2: Explained variance
    ax = axes[1]
    pca_full = PCA()
    pca_full.fit(all_data)
    ax.plot(
        range(1, min(21, len(pca_full.explained_variance_ratio_) + 1)),
        pca_full.explained_variance_ratio_[:20],
        "o-",
        linewidth=2,
    )
    ax.set_xlabel("Principal Component", fontsize=11)
    ax.set_ylabel("Explained Variance Ratio", fontsize=11)
    ax.set_title("Scree Plot", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    filename = f"{save_prefix}pca_comparison_{layer}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Zapisano: {filename}")


def compare_shapes_fixed(triangles_dir, squares_dir, batch_size=16):
    """
    Porównuje oba datasety używając TEGO SAMEGO modelu.
    """

    # Stwórz JEDEN model
    model = SimpleConvNet()
    model.eval()

    print("\n" + "=" * 70)
    print("PORÓWNANIE TRÓJKĄTÓW I KWADRATÓW (ten sam model!)")
    print("=" * 70)
    print(f"\nArchitektura sieci:")
    print(model)

    # Zapisz stan modelu (opcjonalne, dla reprodukowalności)
    torch.save(model.state_dict(), "shared_model_weights.pth")

    # Analizuj oba datasety TYM SAMYM modelem
    print("\n" + "-" * 70)
    print("[1/2] Analizuję trójkąty...")
    tri_act, tri_files = analyze_single_dataset(triangles_dir, model, batch_size)

    print("-" * 70)
    print("[2/2] Analizuję kwadraty...")
    sq_act, sq_files = analyze_single_dataset(squares_dir, model, batch_size)
    print("-" * 70)

    return tri_act, sq_act, tri_files, sq_files, model


# ============================================================================
# 4. MAIN - WORKFLOW
# ============================================================================


def main():
    """
    Główna funkcja porównawcza.
    """
    print("=" * 70)
    print("PORÓWNANIE AKTYWACJI: TRÓJKĄTY vs KWADRATY")
    print("=" * 70)

    # Opcja 1: Załaduj z pickle (jeśli zapisałeś wcześniej)
    # tri_act, tri_files = load_activations('triangles_activations.pkl')
    # sq_act, sq_files = load_activations('squares_activations.pkl')

    # Opcja 2: Uruchom analizy na żywo
    print("\n[1/2] Analizuję trójkąty...")
    from analyze_triangles import analyze_network as analyze_tri

    tri_act, tri_files, tri_model = analyze_tri("triangles_dataset", batch_size=16)

    print("\n[2/2] Analizuję kwadraty...")
    from analyze_squares import analyze_network as analyze_sq

    sq_act, sq_files, sq_model = analyze_sq("squares_dataset", batch_size=16)

    # Zapisz dla przyszłości (opcjonalnie)
    # save_activations(tri_act, tri_files, 'triangles_activations.pkl')
    # save_activations(sq_act, sq_files, 'squares_activations.pkl')

    # ========================================================================
    # ANALIZA STATYSTYCZNA
    # ========================================================================
    print("\n" + "=" * 70)
    print("ANALIZA STATYSTYCZNA")
    print("=" * 70)

    all_layer_metrics = {}
    for layer in ["conv1", "conv2", "conv3", "output"]:
        metrics = compute_selectivity_metrics(tri_act, sq_act, layer)
        all_layer_metrics[layer] = metrics
        print_statistics(metrics, layer)

    # ========================================================================
    # WIZUALIZACJE
    # ========================================================================
    print("\n" + "=" * 70)
    print("TWORZĘ WIZUALIZACJE")
    print("=" * 70)

    # 1. Kompleksowa analiza conv3 (najbardziej interesująca warstwa)
    print("\n[1/4] Kompleksowa analiza conv3...")
    plot_comprehensive_comparison(tri_act, sq_act, layer="conv3", save_prefix="")

    # 2. Podsumowanie wszystkich warstw
    print("\n[2/4] Podsumowanie wszystkich warstw...")
    plot_all_layers_summary(tri_act, sq_act, save_prefix="")

    # 3. PCA dla conv3
    print("\n[3/4] PCA analysis...")
    plot_pca_comparison(tri_act, sq_act, layer="conv3", save_prefix="")

    # 4. Możesz też zrobić dla innych warstw
    print("\n[4/4] Kompleksowa analiza output layer...")
    plot_comprehensive_comparison(tri_act, sq_act, layer="output", save_prefix="")

    # ========================================================================
    # PODSUMOWANIE FINALNE
    # ========================================================================
    print("\n" + "=" * 70)
    print("FINALNE PODSUMOWANIE")
    print("=" * 70)

    for layer in ["conv1", "conv2", "conv3", "output"]:
        m = all_layer_metrics[layer]
        print(f"\n{layer}:")
        print(f"  Korelacja: {m['correlation']:.3f}")
        print(
            f"  Neurony znaczące (p<0.05): {m['significant_neurons']}/{m['total_neurons']} ({m['significant_neurons']/m['total_neurons']*100:.1f}%)"
        )
        print(f"  Duży effect size (|d|>0.8): {(np.abs(m['cohens_d']) > 0.8).sum()}")

    print("\n" + "=" * 70)
    print("GOTOWE! Wygenerowane pliki:")
    print("=" * 70)
    print("  - comprehensive_comparison_conv3.png")
    print("  - comprehensive_comparison_output.png")
    print("  - all_layers_summary.png")
    print("  - pca_comparison_conv3.png")


if __name__ == "__main__":
    main()
