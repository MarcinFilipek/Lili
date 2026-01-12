# unsupervised_learning.py

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
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
# 2. AUTOENCODER (uczenie się bez labeli)
# ============================================================================


class ShapeAutoencoder(nn.Module):
    """
    Uczy się REKONSTRUOWAĆ kształty (bez labeli).
    Musi nauczyć się struktury danych.
    """

    def __init__(self):
        super(ShapeAutoencoder, self).__init__()

        # ENCODER
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=11, stride=2, padding=5),  # 512 -> 256
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=3),  # 256 -> 128
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),  # 128 -> 64
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),  # 64 -> 32
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),  # 32 -> 8
        )

        # LATENT SPACE (tu emerguje reprezentacja!)
        self.latent = nn.Sequential(
            nn.Flatten(), nn.Linear(256 * 8 * 8, 512), nn.ReLU()  # Kompresja do 512
        )

        # DECODER
        self.decoder_fc = nn.Sequential(
            nn.Linear(512, 256 * 8 * 8),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8 -> 16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32 -> 64
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 64 -> 128
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),  # 128 -> 256
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),  # 256 -> 512
            nn.Sigmoid(),  # Wyjście [0, 1]
        )

    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        latent = self.latent(encoded)

        # Decode
        decoded_fc = self.decoder_fc(latent)
        decoded_fc = decoded_fc.view(-1, 256, 8, 8)
        reconstructed = self.decoder(decoded_fc)

        return reconstructed, latent


# ============================================================================
# 3. TRENING AUTOENKODER (bez labeli!)
# ============================================================================


def train_autoencoder(
    model,
    dataloader,
    num_epochs=100,
    device="cpu",
    checkpoint_epochs=[0, 1, 5, 10, 25, 50, 100],
):
    """
    Trenuj autoencoder na MIESZANYCH danych (trójkąty + kwadraty razem, BEZ LABELI).
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()  # Błąd rekonstrukcji

    history = []

    # Zapisz checkpoint epoch 0 (przed treningiem)
    if 0 in checkpoint_epochs:
        torch.save(
            {
                "epoch": 0,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            "unsupervised_checkpoint_epoch_0.pth",
        )
        print("Zapisano checkpoint: epoch 0 (losowe wagi)")

    print(f"\n{'='*70}")
    print("TRENING UNSUPERVISED (autoencoder)")
    print(f"{'='*70}")

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0

        for images, _ in tqdm(
            dataloader, desc=f"Epoch {epoch}/{num_epochs}", leave=False
        ):
            images = images.to(device)

            # Forward
            reconstructed, _ = model(images)
            loss = criterion(reconstructed, images)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        history.append(avg_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{num_epochs}: Loss = {avg_loss:.6f}")

        # Zapisz checkpoint
        if epoch in checkpoint_epochs:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                f"unsupervised_checkpoint_epoch_{epoch}.pth",
            )
            print(f"  → Zapisano checkpoint: epoch {epoch}")

    return history


# ============================================================================
# 4. ANALIZA: CZY EMERGUJE SEPARACJA?
# ============================================================================


def analyze_latent_space(model, tri_loader, sq_loader, device="cpu"):
    """
    Czy w latent space emerguje separacja trójkątów vs kwadratów?
    (Mimo że model NIGDY nie widział labeli!)
    """
    model.eval()

    # Zbierz reprezentacje latent
    tri_latents = []
    sq_latents = []

    with torch.no_grad():
        for images, _ in tqdm(tri_loader, desc="Analizuję trójkąty", leave=False):
            images = images.to(device)
            _, latent = model(images)
            tri_latents.append(latent.cpu().numpy())

        for images, _ in tqdm(sq_loader, desc="Analizuję kwadraty", leave=False):
            images = images.to(device)
            _, latent = model(images)
            sq_latents.append(latent.cpu().numpy())

    tri_latents = np.vstack(tri_latents)
    sq_latents = np.vstack(sq_latents)

    # Oblicz korelację (jak wcześniej)
    tri_mean = tri_latents.mean(axis=0)
    sq_mean = sq_latents.mean(axis=0)
    correlation = np.corrcoef(tri_mean, sq_mean)[0, 1]

    # T-test
    from scipy.stats import ttest_ind

    p_values = []
    for i in range(tri_latents.shape[1]):
        _, p = ttest_ind(tri_latents[:, i], sq_latents[:, i])
        p_values.append(p)
    p_values = np.array(p_values)

    significant = (p_values < 0.05).sum()

    # Cohen's d
    tri_std = tri_latents.std(axis=0)
    sq_std = sq_latents.std(axis=0)
    pooled_std = np.sqrt((tri_std**2 + sq_std**2) / 2)
    cohens_d = (tri_mean - sq_mean) / (pooled_std + 1e-6)
    large_d = (np.abs(cohens_d) > 0.8).sum()
    very_large_d = (np.abs(cohens_d) > 1.2).sum()

    print(f"\n{'='*70}")
    print("ANALIZA LATENT SPACE (unsupervised learning)")
    print(f"{'='*70}")
    print(f"Korelacja: {correlation:.4f}")
    print(f"Neurony znaczące (p<0.05): {significant}/512 ({significant/512*100:.1f}%)")
    print(f"Duży effect size (|d|>0.8): {large_d}/512 ({large_d/512*100:.1f}%)")
    print(
        f"Bardzo duży effect size (|d|>1.2): {very_large_d}/512 ({very_large_d/512*100:.1f}%)"
    )

    return {
        "correlation": correlation,
        "significant_neurons": significant,
        "large_effect_size": large_d,
        "very_large_effect_size": very_large_d,
        "tri_latents": tri_latents,
        "sq_latents": sq_latents,
        "tri_mean": tri_mean,
        "sq_mean": sq_mean,
        "cohens_d": cohens_d,
    }


# ============================================================================
# 5. WIZUALIZACJE
# ============================================================================


def visualize_latent_space(results, save_prefix="unsupervised_"):
    """Wizualizuj latent space PCA."""
    from sklearn.decomposition import PCA

    tri_latents = results["tri_latents"]
    sq_latents = results["sq_latents"]

    # Połącz
    all_latents = np.vstack([tri_latents, sq_latents])

    # PCA
    pca = PCA(n_components=2)
    projected = pca.fit_transform(all_latents)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: PCA scatter
    ax = axes[0]

    tri_proj = projected[: len(tri_latents)]
    sq_proj = projected[len(tri_latents) :]

    ax.scatter(
        tri_proj[:, 0],
        tri_proj[:, 1],
        alpha=0.5,
        s=50,
        c="blue",
        label="Triangles",
        edgecolor="none",
    )
    ax.scatter(
        sq_proj[:, 0],
        sq_proj[:, 1],
        alpha=0.5,
        s=50,
        c="orange",
        label="Squares",
        edgecolor="none",
    )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
    ax.set_title(
        "Latent Space PCA\n(Unsupervised Learning - NO LABELS!)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2)

    # Panel 2: Scatter mean activations
    ax = axes[1]

    tri_mean = results["tri_mean"]
    sq_mean = results["sq_mean"]
    correlation = results["correlation"]

    ax.scatter(tri_mean, sq_mean, alpha=0.5, s=50, c="gray")

    # Linia równej aktywacji
    max_val = max(tri_mean.max(), sq_mean.max())
    ax.plot([0, max_val], [0, max_val], "r--", alpha=0.5, linewidth=2)

    ax.set_xlabel("Mean activation: Triangles", fontsize=12)
    ax.set_ylabel("Mean activation: Squares", fontsize=12)
    ax.set_title(
        f"Neuron Responses\n(r = {correlation:.3f})", fontsize=14, fontweight="bold"
    )
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(f"{save_prefix}latent_space_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Zapisano: {save_prefix}latent_space_analysis.png")


def visualize_reconstructions(
    model, tri_loader, sq_loader, device="cpu", save_prefix="unsupervised_"
):
    """Wizualizuj przykładowe rekonstrukcje."""
    model.eval()

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    # Weź 8 trójkątów i 8 kwadratów
    tri_samples = next(iter(tri_loader))[0][:8].to(device)
    sq_samples = next(iter(sq_loader))[0][:8].to(device)

    with torch.no_grad():
        tri_recon, _ = model(tri_samples)
        sq_recon, _ = model(sq_samples)

    # Plot
    for i in range(4):
        # Trójkąty - oryginał
        axes[i, 0].imshow(tri_samples[i].cpu().permute(1, 2, 0))
        axes[i, 0].set_title("Triangle (original)", fontsize=10)
        axes[i, 0].axis("off")

        # Trójkąty - rekonstrukcja
        axes[i, 1].imshow(tri_recon[i].cpu().permute(1, 2, 0))
        axes[i, 1].set_title("Triangle (reconstructed)", fontsize=10)
        axes[i, 1].axis("off")

        # Kwadraty - oryginał
        axes[i, 2].imshow(sq_samples[i].cpu().permute(1, 2, 0))
        axes[i, 2].set_title("Square (original)", fontsize=10)
        axes[i, 2].axis("off")

        # Kwadraty - rekonstrukcja
        axes[i, 3].imshow(sq_recon[i].cpu().permute(1, 2, 0))
        axes[i, 3].set_title("Square (reconstructed)", fontsize=10)
        axes[i, 3].axis("off")

    plt.suptitle(
        "Autoencoder Reconstructions\n(trained WITHOUT labels!)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(f"{save_prefix}reconstructions.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Zapisano: {save_prefix}reconstructions.png")


def visualize_training_curve(history, save_prefix="unsupervised_"):
    """Wizualizuj krzywą uczenia."""
    plt.figure(figsize=(10, 6))
    plt.plot(history, linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Reconstruction Loss (MSE)", fontsize=12)
    plt.title("Training Curve - Unsupervised Learning", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.2)

    plt.savefig(f"{save_prefix}training_curve.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Zapisano: {save_prefix}training_curve.png")


# ============================================================================
# 6. ANALIZA EMERGENCJI W CZASIE
# ============================================================================


def analyze_emergence(tri_loader, sq_loader, checkpoint_epochs, device="cpu"):
    """
    Analizuj jak separacja emerguje podczas treningu.
    """
    results_over_time = {}

    print(f"\n{'='*70}")
    print("ANALIZA EMERGENCJI SEPARACJI W CZASIE")
    print(f"{'='*70}")

    for epoch in checkpoint_epochs:
        print(f"\nŁadowanie checkpointu epoch {epoch}...")

        # Załaduj model
        model = ShapeAutoencoder()
        checkpoint = torch.load(f"unsupervised_checkpoint_epoch_{epoch}.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        # Analizuj
        results = analyze_latent_space(model, tri_loader, sq_loader, device)
        results_over_time[epoch] = results

    # Wizualizuj emergencję
    print(f"\n{'='*70}")
    print("PODSUMOWANIE EMERGENCJI")
    print(f"{'='*70}")

    epochs = sorted(results_over_time.keys())
    correlations = [results_over_time[e]["correlation"] for e in epochs]
    significant = [results_over_time[e]["significant_neurons"] for e in epochs]
    large_d = [results_over_time[e]["large_effect_size"] for e in epochs]

    # Tabela
    print(f"\n{'Epoch':<10} {'Correlation':<15} {'Significant':<20} {'Large d':<15}")
    print("-" * 60)
    for i, epoch in enumerate(epochs):
        print(
            f"{epoch:<10} {correlations[i]:<15.4f} {significant[i]:<20} {large_d[i]:<15}"
        )

    # Wykres
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Korelacja
    ax = axes[0]
    ax.plot(epochs, correlations, "o-", linewidth=2, markersize=8)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Correlation", fontsize=12)
    ax.set_title("Correlation over Training", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="r=0.5")
    ax.legend()

    # Panel 2: Neurony znaczące
    ax = axes[1]
    ax.plot(epochs, significant, "o-", linewidth=2, markersize=8, color="green")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Significant Neurons", fontsize=12)
    ax.set_title("Significant Neurons over Training", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2)

    # Panel 3: Duży effect size
    ax = axes[2]
    ax.plot(epochs, large_d, "o-", linewidth=2, markersize=8, color="purple")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Neurons with |d| > 0.8", fontsize=12)
    ax.set_title("Large Effect Size over Training", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig("unsupervised_emergence_over_time.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\nZapisano: unsupervised_emergence_over_time.png")

    return results_over_time


# ============================================================================
# 7. MAIN
# ============================================================================

if __name__ == "__main__":
    # Parametry
    TRIANGLES_DIR = "triangles_dataset"
    SQUARES_DIR = "squares_dataset"
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    CHECKPOINT_EPOCHS = [0, 1, 5, 10, 25, 50, 100]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"{'='*70}")
    print("UNSUPERVISED LEARNING EXPERIMENT")
    print(f"{'='*70}")
    print(f"Urządzenie: {DEVICE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")

    # Sprawdź czy foldery istnieją
    if not os.path.exists(TRIANGLES_DIR):
        print(f"\nBŁĄD: Folder {TRIANGLES_DIR} nie istnieje!")
        exit()

    if not os.path.exists(SQUARES_DIR):
        print(f"\nBŁĄD: Folder {SQUARES_DIR} nie istnieje!")
        exit()

    # Transformacje
    transform = transforms.Compose([transforms.ToTensor()])

    # Datasety
    tri_dataset = ShapeDataset(TRIANGLES_DIR, transform=transform)
    sq_dataset = ShapeDataset(SQUARES_DIR, transform=transform)

    print(f"\nTrójkąty: {len(tri_dataset)} obrazów")
    print(f"Kwadraty: {len(sq_dataset)} obrazów")

    # Połącz w jeden dataset (BEZ ROZRÓŻNIANIA!)
    combined_dataset = ConcatDataset([tri_dataset, sq_dataset])
    combined_loader = DataLoader(
        combined_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )

    # Osobne loadery dla testowania
    tri_loader = DataLoader(
        tri_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )
    sq_loader = DataLoader(
        sq_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    print(f"Całkowita liczba obrazów (mixed): {len(combined_dataset)}")

    # Model
    model = ShapeAutoencoder()
    print(f"\nLiczba parametrów: {sum(p.numel() for p in model.parameters()):,}")

    # TRENING (bez labeli!)
    history = train_autoencoder(
        model,
        combined_loader,
        num_epochs=NUM_EPOCHS,
        device=DEVICE,
        checkpoint_epochs=CHECKPOINT_EPOCHS,
    )

    # Zapisz finalny model
    torch.save(model.state_dict(), "autoencoder_unsupervised_final.pth")
    print("\nZapisano: autoencoder_unsupervised_final.pth")

    # Wizualizuj training curve
    visualize_training_curve(history)

    # ANALIZA FINALNA: czy emerguje separacja?
    print(f"\n{'='*70}")
    print("ANALIZA FINALNA (po treningu)")
    print(f"{'='*70}")

    results = analyze_latent_space(model, tri_loader, sq_loader, device=DEVICE)

    # Wizualizacje
    visualize_latent_space(results)
    visualize_reconstructions(model, tri_loader, sq_loader, device=DEVICE)

    # ANALIZA EMERGENCJI W CZASIE
    emergence_results = analyze_emergence(
        tri_loader, sq_loader, CHECKPOINT_EPOCHS, device=DEVICE
    )

    # Finalne podsumowanie
    print(f"\n{'='*70}")
    print("FINALNE PODSUMOWANIE")
    print(f"{'='*70}")

    final_corr = results["correlation"]
    final_sig = results["significant_neurons"]
    final_large_d = results["large_effect_size"]

    print(f"\nWyniki po {NUM_EPOCHS} epochs unsupervised learning:")
    print(f"  Korelacja: {final_corr:.4f}")
    print(f"  Neurony znaczące: {final_sig}/512 ({final_sig/512*100:.1f}%)")
    print(f"  Duży effect size: {final_large_d}/512 ({final_large_d/512*100:.1f}%)")

    print("\n" + "=" * 70)
    print("INTERPRETACJA:")
    print("=" * 70)

    if final_corr < 0.7:
        print("✓✓ SILNA SEPARACJA emerguje z unsupervised learning!")
        print(
            "   → Struktura danych wizualnych SAMA W SOBIE niesie informację o kształtach"
        )
        print("   → To WSPIERA tezę że noworodek nie potrzebuje supervision")
    elif final_corr < 0.85:
        print("✓ UMIARKOWANA SEPARACJA emerguje")
        print("   → Częściowo wspiera tezę o unsupervised learning")
        print("   → Pewna struktura emerguje, ale słabsza niż supervised")
    else:
        print("❌ SŁABA lub BRAK SEPARACJI")
        print("   → Unsupervised autoencoder nie wystarczy")
        print("   → Potrzebne inne mechanizmy (Hebbian? Predictive coding?)")

    print("\n" + "=" * 70)
    print("WYGENEROWANE PLIKI:")
    print("=" * 70)
    print("  - autoencoder_unsupervised_final.pth")
    print("  - unsupervised_checkpoint_epoch_X.pth (dla różnych X)")
    print("  - unsupervised_latent_space_analysis.png")
    print("  - unsupervised_reconstructions.png")
    print("  - unsupervised_training_curve.png")
    print("  - unsupervised_emergence_over_time.png")

    print("\n" + "=" * 70)
    print("GOTOWE!")
    print("=" * 70)
