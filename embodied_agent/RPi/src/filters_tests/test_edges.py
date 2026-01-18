"""
Test edges feature extraction.
Shows Sobel edge detection in 8x8 grid.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


def extract_edges(gray_frame):
    """
    Extract edges in 8×8 grid using Sobel operator.
    
    Args:
        gray_frame: grayscale image (H, W)
        
    Returns:
        edges: np.array([64]) - flattened 8x8 grid
    """
    # Sobel edge detection
    sobelx = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)
    
    # Magnitude
    edges = np.sqrt(sobelx**2 + sobely**2)
    if edges.max() > 0:
        edges = edges / edges.max()
    
    # Resize to 8×8
    edges_small = cv2.resize(edges, (8, 8), interpolation=cv2.INTER_AREA)
    
    return edges_small.flatten()  # shape: (64,)


def visualize_edges(gray_frame):
    """Visualize edge extraction"""
    # Extract features
    edges_flat = extract_edges(gray_frame)
    edges_grid = edges_flat.reshape(8, 8)
    
    # Also compute full-resolution edges for visualization
    sobelx = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)
    edges_full = np.sqrt(sobelx**2 + sobely**2)
    if edges_full.max() > 0:
        edges_full = edges_full / edges_full.max()
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Original image
    axes[0, 0].imshow(gray_frame, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Full-resolution edges
    axes[0, 1].imshow(edges_full, cmap='hot')
    axes[0, 1].set_title('Sobel Edge Detection (Full Resolution)')
    axes[0, 1].axis('off')
    
    # 8x8 grid overlay on edges
    h, w = edges_full.shape
    axes[1, 0].imshow(edges_full, cmap='hot')
    
    # Draw grid lines
    for i in range(1, 8):
        axes[1, 0].axhline(y=i * h // 8, color='cyan', linewidth=1)
        axes[1, 0].axvline(x=i * w // 8, color='cyan', linewidth=1)
    
    axes[1, 0].set_title('8×8 Grid Overlay')
    axes[1, 0].axis('off')
    
    # Edge strength heatmap
    im = axes[1, 1].imshow(edges_grid, cmap='hot', vmin=0, vmax=1)
    axes[1, 1].set_title('Edge Features (8×8)')
    
    # Add grid
    for i in range(9):
        axes[1, 1].axhline(y=i-0.5, color='white', linewidth=0.5, alpha=0.3)
        axes[1, 1].axvline(x=i-0.5, color='white', linewidth=0.5, alpha=0.3)
    
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/edges_test.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: edges_test.png")
    plt.show()
    
    # Print feature vector
    print("\nEdges feature vector (64 dims):")
    print(edges_flat)
    print(f"Shape: {edges_flat.shape}")
    print(f"Min: {edges_flat.min():.3f}, Max: {edges_flat.max():.3f}")
    print(f"Mean edge strength: {edges_flat.mean():.3f}")
    print(f"Non-zero cells: {np.sum(edges_flat > 0.1)} / 64")


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
        # Create test image with edges
        gray = np.zeros((480, 640), dtype=np.uint8)
        cv2.rectangle(gray, (100, 100), (300, 300), 255, -1)
        cv2.circle(gray, (450, 240), 100, 128, -1)
        print("Using synthetic test image with shapes")
    
    # Visualize
    visualize_edges(gray)
    
    print("\n" + "="*50)
    print("EDGES FEATURE EXPLANATION:")
    print("="*50)
    print("- Uses Sobel operator to detect edges")
    print("- Combines horizontal (∂x) and vertical (∂y) gradients")
    print("- Downsampled to 8×8 = 64 cells")
    print("- Each cell = edge strength [0, 1]")
    print("- Bio-inspiration: V1 simple cells (orientation-selective)")
    print("- Use: Spatial structure, object boundaries")
