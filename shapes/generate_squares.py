# generate_squares.py

import numpy as np
from PIL import Image, ImageDraw
import random
import os


def generate_square(size=512, shape_size=None, position=None, rotation=0):
    """
    Generuje obraz z czarnym wypełnionym kwadratem na białym tle.

    Args:
        size: rozmiar obrazu (512x512)
        shape_size: wielkość kwadratu (bok w pikselach), jeśli None to losowa
        position: pozycja środka (x, y), jeśli None to losowa
        rotation: obrót w stopniach (0-360)

    Returns:
        PIL Image
    """
    # Losuj parametry jeśli nie podane
    if shape_size is None:
        shape_size = random.randint(50, 300)  # wielkość boku kwadratu

    if position is None:
        # Upewnij się że kwadrat nie wychodzi poza obrazek
        margin = int(
            shape_size * 0.8
        )  # margin sufficient for rotation (sqrt(2)/2 ~ 0.707)
        position = (
            random.randint(margin, size - margin),
            random.randint(margin, size - margin),
        )

    # Stwórz białe tło
    img = Image.new("RGB", (size, size), color="white")
    draw = ImageDraw.Draw(img)

    # Oblicz wierzchołki kwadratu
    # Środek w position, wierzchołki względem środka
    half_size = shape_size / 2

    # Wierzchołki przed obrotem (kwadrat z bokami równoległymi do osi)
    vertices_centered = np.array(
        [
            [-half_size, -half_size],  # lewy górny
            [half_size, -half_size],  # prawy górny
            [half_size, half_size],  # prawy dolny
            [-half_size, half_size],  # lewy dolny
        ]
    )

    # Macierz rotacji
    angle_rad = np.radians(rotation)
    rotation_matrix = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )

    # Obróć wierzchołki
    vertices_rotated = vertices_centered @ rotation_matrix.T

    # Przesuń do docelowej pozycji
    vertices_final = vertices_rotated + np.array(position)

    # Konwertuj do listy tupli dla PIL
    vertices_list = [tuple(v) for v in vertices_final]

    # Narysuj wypełniony kwadrat
    draw.polygon(vertices_list, fill="black", outline="black")

    return img


def generate_square_dataset(n_images=100, output_dir="squares", size=512):
    """
    Generuje dataset z kwadratami.

    Args:
        n_images: liczba obrazów do wygenerowania
        output_dir: folder wyjściowy
        size: rozmiar obrazu
    """
    # Stwórz folder jeśli nie istnieje
    os.makedirs(output_dir, exist_ok=True)

    for i in range(n_images):
        # Losowe parametry
        shape_size = random.randint(50, 300)
        margin = int(shape_size * 0.8)
        position = (
            random.randint(margin, size - margin),
            random.randint(margin, size - margin),
        )
        rotation = random.randint(0, 360)

        # Generuj obraz
        img = generate_square(
            size=size, shape_size=shape_size, position=position, rotation=rotation
        )

        # Zapisz
        filename = f"square_{i:04d}_size{shape_size}_rot{rotation}.png"
        img.save(os.path.join(output_dir, filename))

        if (i + 1) % 10 == 0:
            print(f"Wygenerowano {i + 1}/{n_images} obrazów")

    print(f"Gotowe! Wszystkie obrazy w folderze: {output_dir}")


def show_examples(n_examples=9):
    """
    Pokaż przykładowe obrazy w gridzie.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        # Losowe parametry
        img = generate_square(
            size=512,
            shape_size=random.randint(50, 300),
            position=(random.randint(100, 412), random.randint(100, 412)),
            rotation=random.randint(0, 360),
        )

        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("square_examples.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # Pokaż przykłady
    print("Generuję przykładowe obrazy kwadratów...")
    show_examples(n_examples=9)

    # Generuj dataset
    print("\nGeneruję pełny dataset kwadratów...")
    generate_square_dataset(n_images=100, output_dir="squares_dataset", size=512)
