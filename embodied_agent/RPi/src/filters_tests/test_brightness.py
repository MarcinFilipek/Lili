"""
Test brightness feature extraction.
Shows how brightness is extracted as 128x128 grid.
"""
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend (for headless Raspberry Pi)
import matplotlib.pyplot as plt


def extract_brightness(gray_frame):
    """
    Extract brightness in 128×128 grid.
    
    Args:
        gray_frame: grayscale image (H, W)
        
    Returns:
        brightness: np.array([16384]) - flattened 128x128 grid
    """
    h, w = gray_frame.shape
    grid_size = 128
    cell_h, cell_w = h // grid_size, w // grid_size
    
    brightness = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            cell = gray_frame[i*cell_h:(i+1)*cell_h,
                            j*cell_w:(j+1)*cell_w]
            brightness[i, j] = np.mean(cell) / 255.0
    
    return brightness.flatten()  # shape: (16384,)


def visualize_brightness(gray_frame):
    """Visualize brightness extraction"""
    brightness_flat = extract_brightness(gray_frame)
    brightness_grid = brightness_flat.reshape(128, 128)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(gray_frame, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Grid overlay — rysujemy co 16 linii żeby nie zaśmiecić wizualizacji
    h, w = gray_frame.shape
    axes[1].imshow(gray_frame, cmap='gray')
    for i in range(1, 128):
        if i % 16 == 0:
            axes[1].axhline(y=i * h // 128, color='red', linewidth=0.5)
            axes[1].axvline(x=i * w // 128, color='red', linewidth=0.5)
    axes[1].set_title('128×128 Grid Overlay (co 16)')
    axes[1].axis('off')
    
    # Brightness heatmap
    im = axes[2].imshow(brightness_grid, cmap='hot', vmin=0, vmax=1)
    axes[2].set_title('Brightness Features (128×128)')
    axes[2].axis('off')
    
    plt.colorbar(im, ax=axes[2])
    plt.tight_layout()
    plt.savefig('brightness_test.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: brightness_test.png")
    plt.close()
    
    print("\nBrightness feature vector (16384 dims):")
    print(f"  Shape: {brightness_flat.shape}")
    print(f"  Min: {brightness_flat.min():.3f}  Max: {brightness_flat.max():.3f}  Mean: {brightness_flat.mean():.3f}")


if __name__ == "__main__":
    try:
        from picamera2 import Picamera2
        
        print("Opening Picamera2...")
        width = 640
        height = 480
        
        picam = Picamera2()
        cam_conf = picam.create_video_configuration(
            main={"format": "RGB888", "size": (width, height)}
        )
        picam.configure(cam_conf)
        picam.start()
        
        frame = picam.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        picam.stop()
        picam.close()
        print("✓ Captured frame from Picamera2")
        
    except Exception as e:
        print(f"✗ Picamera2 not available: {e}")
        gray = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        print("Using random test image")
    
    visualize_brightness(gray)
    
    print("\n" + "="*50)
    print("BRIGHTNESS FEATURE EXPLANATION:")
    print("="*50)
    print("- Divides image into 128×128 = 16384 cells")
    print("- Each cell = average brightness [0, 1]")
    print("- Output: 16384-dimensional feature vector")
    print("- Bio-inspiration: Retinal ganglion cells")
    print("- Use: Basic light/dark pattern detection")