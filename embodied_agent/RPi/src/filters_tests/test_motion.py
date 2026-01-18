"""
Test motion feature extraction.
Shows optical flow in 4x4 grid.
Requires two consecutive frames to compute motion.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


def extract_motion(current_frame, previous_frame):
    """
    Extract optical flow in 4×4 grid.
    
    Args:
        current_frame: grayscale image (H, W)
        previous_frame: grayscale image (H, W)
        
    Returns:
        motion: np.array([32]) - flattened 4x4x2 (dx, dy for each cell)
    """
    if previous_frame is None:
        return np.zeros(32)
    
    # Dense optical flow (Farneback algorithm)
    flow = cv2.calcOpticalFlowFarneback(
        previous_frame, current_frame,
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    
    # Resize flow to 4×4
    flow_small = cv2.resize(flow, (4, 4), interpolation=cv2.INTER_AREA)
    
    # Normalize
    flow_magnitude = np.sqrt(flow_small[:,:,0]**2 + flow_small[:,:,1]**2)
    if flow_magnitude.max() > 0:
        flow_small = flow_small / flow_magnitude.max()
    
    return flow_small.flatten()  # shape: (32,) - [dx0, dy0, dx1, dy1, ...]


def visualize_motion(current_frame, previous_frame):
    """Visualize motion extraction"""
    # Extract features
    motion_flat = extract_motion(current_frame, previous_frame)
    motion_grid = motion_flat.reshape(4, 4, 2)
    
    # Compute full-resolution flow for visualization
    flow = cv2.calcOpticalFlowFarneback(
        previous_frame, current_frame,
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    
    # Create flow visualization (HSV)
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2  # Hue = direction
    hsv[..., 1] = 255  # Saturation = full
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value = magnitude
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # Create visualization
    fig = plt.figure(figsize=(16, 12))
    
    # Previous frame
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(previous_frame, cmap='gray')
    ax1.set_title('Previous Frame (t-1)')
    ax1.axis('off')
    
    # Current frame
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(current_frame, cmap='gray')
    ax2.set_title('Current Frame (t)')
    ax2.axis('off')
    
    # Flow visualization (color = direction, brightness = magnitude)
    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(flow_rgb)
    ax3.set_title('Optical Flow (Full Resolution)\nColor=Direction, Brightness=Speed')
    ax3.axis('off')
    
    # Flow magnitude
    ax4 = plt.subplot(2, 3, 4)
    magnitude = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
    im1 = ax4.imshow(magnitude, cmap='hot')
    ax4.set_title('Flow Magnitude')
    ax4.axis('off')
    plt.colorbar(im1, ax=ax4)
    
    # 4x4 grid with arrows
    ax5 = plt.subplot(2, 3, 5)
    ax5.imshow(current_frame, cmap='gray', alpha=0.3)
    
    # Draw arrows for each cell
    cell_h, cell_w = h // 4, w // 4
    for i in range(4):
        for j in range(4):
            cx = (j + 0.5) * cell_w
            cy = (i + 0.5) * cell_h
            dx = motion_grid[i, j, 0] * cell_w
            dy = motion_grid[i, j, 1] * cell_h
            
            # Draw arrow
            ax5.arrow(cx, cy, dx*2, dy*2, 
                     head_width=20, head_length=20, 
                     fc='red', ec='red', linewidth=2)
    
    # Draw grid
    for i in range(1, 4):
        ax5.axhline(y=i * h // 4, color='cyan', linewidth=1)
        ax5.axvline(x=i * w // 4, color='cyan', linewidth=1)
    
    ax5.set_title('4×4 Flow Vectors (Red Arrows)')
    ax5.axis('off')
    
    # Flow magnitude heatmap 4x4
    ax6 = plt.subplot(2, 3, 6)
    flow_magnitude_grid = np.sqrt(motion_grid[:,:,0]**2 + motion_grid[:,:,1]**2)
    im2 = ax6.imshow(flow_magnitude_grid, cmap='hot', vmin=0, vmax=1)
    ax6.set_title('Motion Magnitude (4×4)')
    
    # Add text values
    for i in range(4):
        for j in range(4):
            mag = flow_magnitude_grid[i, j]
            text = ax6.text(j, i, f'{mag:.2f}',
                          ha="center", va="center", 
                          color="white" if mag > 0.5 else "black", 
                          fontsize=10)
    
    plt.colorbar(im2, ax=ax6)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/motion_test.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: motion_test.png")
    plt.show()
    
    # Print feature vector
    print("\nMotion feature vector (32 dims = 4×4×2):")
    print(motion_flat)
    print(f"Shape: {motion_flat.shape}")
    print(f"Min: {motion_flat.min():.3f}, Max: {motion_flat.max():.3f}")
    print(f"Average motion magnitude: {flow_magnitude_grid.mean():.3f}")


if __name__ == "__main__":
    print("Opening webcam for motion detection...")
    print("Move something in front of the camera or move the camera!")
    
    cap = cv2.VideoCapture(0)
    
    if cap.isOpened():
        print("✓ Webcam opened")
        print("Capturing frames...")
        
        # Wait a bit for camera to adjust
        time.sleep(0.5)
        
        # Capture first frame
        ret1, frame1 = cap.read()
        if not ret1:
            print("✗ Failed to capture first frame")
            cap.release()
            exit(1)
        
        prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        
        # Wait a bit and capture second frame
        time.sleep(0.2)
        
        ret2, frame2 = cap.read()
        if not ret2:
            print("✗ Failed to capture second frame")
            cap.release()
            exit(1)
        
        curr_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        cap.release()
        print("✓ Captured two consecutive frames")
        
    else:
        print("✗ Webcam not available")
        print("Creating synthetic motion example...")
        
        # Create two frames with simulated motion
        prev_gray = np.zeros((480, 640), dtype=np.uint8)
        curr_gray = np.zeros((480, 640), dtype=np.uint8)
        
        # Moving rectangle
        cv2.rectangle(prev_gray, (200, 200), (400, 400), 255, -1)
        cv2.rectangle(curr_gray, (220, 210), (420, 410), 255, -1)  # moved right and down
        
        # Moving circle
        cv2.circle(prev_gray, (100, 100), 50, 128, -1)
        cv2.circle(curr_gray, (100, 150), 50, 128, -1)  # moved down
        
        print("Using synthetic frames with simulated motion")
    
    # Visualize
    visualize_motion(curr_gray, prev_gray)
    
    print("\n" + "="*50)
    print("MOTION FEATURE EXPLANATION:")
    print("="*50)
    print("- Uses Farneback optical flow algorithm")
    print("- Computes motion vectors (dx, dy) for each pixel")
    print("- Downsampled to 4×4 = 16 cells, each with (dx, dy)")
    print("- Total: 32 dimensions (16 cells × 2 components)")
    print("- Bio-inspiration: MT cortex (motion processing)")
    print("- Use: Ego-motion detection, object tracking")
    print("\nNOTE: Motion requires TWO frames to compute!")
