"""
Test brightness feature extraction.
Shows how brightness is extracted as 4x4 grid.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


def extract_brightness(gray_frame):
    """
    Extract brightness in 4×4 grid.
    
    Args:
        gray_frame: grayscale image (H, W)
        
    Returns:
        brightness: np.array([16]) - flattened 4x4 grid
    """
    h, w = gray_frame.shape
    grid_size = 4
    cell_h, cell_w = h // grid_size, w // grid_size
    
    brightness = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            cell = gray_frame[i*cell_h:(i+1)*cell_h,
                            j*cell_w:(j+1)*cell_w]
            brightness[i, j] = np.mean(cell) / 255.0
    
    return brightness.flatten()  # shape: (16,)


def visualize_brightness(gray_frame):
    """Visualize brightness extraction"""
    # Extract features
    brightness_flat = extract_brightness(gray_frame)
    brightness_grid = brightness_flat.reshape(4, 4)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(gray_frame, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 4x4 grid overlay
    h, w = gray_frame.shape
    axes[1].imshow(gray_frame, cmap='gray')
    
    # Draw grid lines
    for i in range(1, 4):
        axes[1].axhline(y=i * h // 4, color='red', linewidth=2)
        axes[1].axvline(x=i * w // 4, color='red', linewidth=2)
    
    axes[1].set_title('4×4 Grid Overlay')
    axes[1].axis('off')
    
    # Brightness heatmap
    im = axes[2].imshow(brightness_grid, cmap='hot', vmin=0, vmax=1)
    axes[2].set_title('Brightness Features (4×4)')
    
    # Add text values
    for i in range(4):
        for j in range(4):
            text = axes[2].text(j, i, f'{brightness_grid[i, j]:.2f}',
                              ha="center", va="center", color="white", fontsize=12)
    
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/brightness_test.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: brightness_test.png")
    plt.show()
    
    # Print feature vector
    print("\nBrightness feature vector (16 dims):")
    print(brightness_flat)
    print(f"Shape: {brightness_flat.shape}")
    print(f"Min: {brightness_flat.min():.3f}, Max: {brightness_flat.max():.3f}")


if __name__ == "__main__":
    # Try to use Picamera2
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
        
        # Capture frame
        frame = picam.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        picam.stop()
        picam.close()
        print("✓ Captured frame from Picamera2")
        
    except Exception as e:
        print(f"✗ Picamera2 not available: {e}")
        # Fallback to test image
        gray = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        print("Using random test image")
    
    # Visualize
    visualize_brightness(gray)
    
    print("\n" + "="*50)
    print("BRIGHTNESS FEATURE EXPLANATION:")
    print("="*50)
    print("- Divides image into 4×4 = 16 cells")
    print("- Each cell = average brightness [0, 1]")
    print("- Output: 16-dimensional feature vector")
    print("- Bio-inspiration: Retinal ganglion cells")
    print("- Use: Basic light/dark pattern detection")
