"""
Test corners feature extraction.
Shows Harris corner detection in 6x6 grid.
"""
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend (for headless Raspberry Pi)
import matplotlib.pyplot as plt


def extract_corners(gray_frame):
    """
    Extract corners in 6×6 grid using Harris corner detector.
    
    Args:
        gray_frame: grayscale image (H, W)
        
    Returns:
        corners: np.array([36]) - flattened 6x6 grid
    """
    # Convert to float32 for Harris
    gray_float = np.float32(gray_frame)
    
    # Harris corner detection
    corners = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)
    
    # Normalize
    if corners.max() > 0:
        corners = corners / corners.max()
    
    # Resize to 6×6
    corners_small = cv2.resize(corners, (6, 6), interpolation=cv2.INTER_AREA)
    
    return corners_small.flatten()  # shape: (36,)


def visualize_corners(gray_frame):
    """Visualize corner extraction"""
    # Extract features
    corners_flat = extract_corners(gray_frame)
    corners_grid = corners_flat.reshape(6, 6)
    
    # Compute full-resolution corners for visualization
    gray_float = np.float32(gray_frame)
    corners_full = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)
    
    # Normalize for display
    corners_norm = corners_full.copy()
    if corners_norm.max() > 0:
        corners_norm = corners_norm / corners_norm.max()
    
    # Create corner overlay (dilate for visibility)
    corners_display = np.zeros_like(gray_frame)
    corners_display[corners_full > 0.01 * corners_full.max()] = 255
    corners_display = cv2.dilate(corners_display, None)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Original image
    axes[0, 0].imshow(gray_frame, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Corners overlaid on image
    axes[0, 1].imshow(gray_frame, cmap='gray')
    
    # Find corner points
    corner_coords = np.where(corners_full > 0.01 * corners_full.max())
    axes[0, 1].plot(corner_coords[1], corner_coords[0], 'r.', markersize=3)
    axes[0, 1].set_title('Harris Corners Detected (Red Dots)')
    axes[0, 1].axis('off')
    
    # Corner strength heatmap (full resolution)
    axes[1, 0].imshow(corners_norm, cmap='hot')
    
    # Draw 6x6 grid overlay
    h, w = corners_norm.shape
    for i in range(1, 6):
        axes[1, 0].axhline(y=i * h // 6, color='cyan', linewidth=1)
        axes[1, 0].axvline(x=i * w // 6, color='cyan', linewidth=1)
    
    axes[1, 0].set_title('Corner Strength Heatmap + 6×6 Grid')
    axes[1, 0].axis('off')
    
    # Corner strength heatmap 6x6
    im = axes[1, 1].imshow(corners_grid, cmap='hot', vmin=0, vmax=1)
    axes[1, 1].set_title('Corner Features (6×6)')
    
    # Add grid lines
    for i in range(7):
        axes[1, 1].axhline(y=i-0.5, color='white', linewidth=0.5, alpha=0.3)
        axes[1, 1].axvline(x=i-0.5, color='white', linewidth=0.5, alpha=0.3)
    
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/corners_test.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: corners_test.png")
    plt.close()  # Close figure instead of show()
    
    # Print feature vector
    print("\nCorners feature vector (36 dims):")
    print(corners_flat)
    print(f"Shape: {corners_flat.shape}")
    print(f"Min: {corners_flat.min():.3f}, Max: {corners_flat.max():.3f}")
    print(f"Mean corner strength: {corners_flat.mean():.3f}")
    print(f"Strong corners (>0.3): {np.sum(corners_flat > 0.3)} / 36")
    
    # Identify which cells have strong corners
    strong_corners = np.where(corners_grid > 0.3)
    if len(strong_corners[0]) > 0:
        print(f"\nCells with strong corners:")
        for i, j in zip(strong_corners[0], strong_corners[1]):
            print(f"  Row {i}, Col {j}: strength = {corners_grid[i,j]:.3f}")


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
        # Create test image with clear corners
        gray = np.zeros((480, 640), dtype=np.uint8)
        # Multiple shapes with corners
        cv2.rectangle(gray, (100, 100), (300, 300), 255, 3)
        cv2.line(gray, (400, 100), (400, 250), 200, 3)
        cv2.line(gray, (400, 250), (550, 250), 200, 3)
        cv2.line(gray, (200, 350), (400, 350), 150, 3)
        cv2.line(gray, (300, 280), (300, 420), 150, 3)
        print("Using synthetic test image with corners")
    
    # Visualize
    visualize_corners(gray)
    
    print("\n" + "="*50)
    print("CORNERS FEATURE EXPLANATION:")
    print("="*50)
    print("- Uses Harris corner detector")
    print("- Detects points where gradients change in multiple directions")
    print("- Downsampled to 6×6 = 36 cells")
    print("- Each cell = corner strength [0, 1]")
    print("- Bio-inspiration: V1 complex cells (junction detectors)")
    print("- Use: Object landmarks, spatial localization")
    print("\nCorners are good for:")
    print("  • Identifying object vertices (boxes, furniture)")
    print("  • Texture-rich regions")
    print("  • Distinctive spatial features for navigation")
