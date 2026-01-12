import numpy as np
from PIL import Image, ImageDraw
import random
import os


def generate_equilateral_triangle(size=512, shape_size=None, position=None, rotation=0):
    """
    Generuje obraz z czarnym wypełnionym trójkątem równobocznym na białym tle.

    Args:
        size: rozmiar obrazu (512x512)
        shape_size: wielkość trójkąta (bok w pikselach), jeśli None to losowa
        position: pozycja środka (x, y), jeśli None to losowa
        rotation: obrót w stopniach (0-360), jeśli 0 i chcemy losowy to trzeba podać wartość

    Returns:
        PIL Image
    """
    # Losuj parametry jeśli nie podane
    if shape_size is None:
        shape_size = random.randint(50, 300)  # wielkość boku trójkąta

    if position is None:
        # Upewnij się że trójkąt nie wychodzi poza obrazek
        # Margines to przybliżony promień okręgu opisanego (0.577 * a) + zapas
        margin = int(shape_size * 0.6) + 20
        position = (
            random.randint(margin, size - margin),
            random.randint(margin, size - margin),
        )

    # Stwórz białe tło
    img = Image.new("RGB", (size, size), color="white")
    draw = ImageDraw.Draw(img)

    # Oblicz wierzchołki trójkąta równobocznego
    # Środek w (0, 0), wierzchołki względem środka
    height = shape_size * np.sqrt(3) / 2  # wysokość trójkąta równobocznego

    # Wierzchołki przed obrotem (trójkąt "stoi" na podstawie)
    vertices_centered = np.array(
        [
            [0, -2 * height / 3],  # górny wierzchołek
            [-shape_size / 2, height / 3],  # lewy dolny
            [shape_size / 2, height / 3],  # prawy dolny
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

    # Narysuj wypełniony trójkąt
    draw.polygon(vertices_list, fill="black", outline="black")

    return img


def generate_triangle_dataset(n_images=100, output_dir="triangles", size=512):
    """
    Generuje dataset z trójkątami równobocznymi.

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
        margin = int(shape_size * 0.6) + 20
        position = (
            random.randint(margin, size - margin),
            random.randint(margin, size - margin),
        )
        rotation = random.randint(0, 360)

        # Generuj obraz
        img = generate_equilateral_triangle(
            size=size, shape_size=shape_size, position=position, rotation=rotation
        )

        # Zapisz
        filename = f"triangle_{i:04d}_size{shape_size}_rot{rotation}.png"
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
        img = generate_equilateral_triangle(
            size=512,
            shape_size=random.randint(50, 300),
            position=(random.randint(100, 412), random.randint(100, 412)),
            rotation=random.randint(0, 360),
        )

        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("triangle_examples.png", dpi=150, bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    # Pokaż przykłady
    print("Generuję przykładowe obrazy...")
    show_examples(n_examples=9)

    # Generuj dataset
    print("\nGeneruję pełny dataset...")
    generate_triangle_dataset(n_images=100, output_dir="triangles_dataset", size=512)

    # Możesz też wygenerować pojedynczy obraz:
    # img = generate_equilateral_triangle()
    # img.save('test_triangle.png')
