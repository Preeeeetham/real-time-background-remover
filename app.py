import cv2
import numpy as np
import os
from datetime import datetime
import time

def capture_image():
    """Capture an image from the camera"""
    print("Starting camera... Press SPACE to capture or ESC to quit")
    
    cap = cv2.VideoCapture(0)  # Open default camera (usually webcam)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None
    
    # Wait for camera to initialize
    for _ in range(10):
        cap.read()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Show live preview
        cv2.imshow('Press SPACE to capture, ESC to quit', frame)
        
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
        elif key == 32:  # SPACE key
            cap.release()
            cv2.destroyAllWindows()
            return frame
    
    # Release camera and close windows if loop exits
    cap.release()
    cv2.destroyAllWindows()
    return None

def remove_background(image):
    """Remove background using GrabCut algorithm with automatic initialization"""
    # Convert to RGB for processing (GrabCut works better with RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create a mask initialized with zeros
    mask = np.zeros(image.shape[:2], np.uint8)
    
    # Create temporary arrays for the models
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Define initial rectangle to start GrabCut
    # We'll use a slightly smaller rectangle than the full image
    h, w = image.shape[:2]
    margin = min(h, w) // 8  # Use 1/8 of the smaller dimension as margin
    rect = (margin, margin, w - 2*margin, h - 2*margin)
    
    # Run GrabCut algorithm
    print("Processing image with GrabCut (this may take a moment)...")
    cv2.grabCut(image_rgb, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    # Create a mask where background is 0 and foreground is 1
    # GrabCut assigns 0 (sure background) and 2 (probable background) to background
    # and 1 (sure foreground) and 3 (probable foreground) to foreground
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # Apply the mask to get foreground
    foreground = image.copy()
    foreground = cv2.bitwise_and(foreground, foreground, mask=mask2)
    
    # Extract background (original - foreground)
    background = image.copy()
    inv_mask = cv2.bitwise_not(mask2)
    background = cv2.bitwise_and(background, background, mask=inv_mask)
    
    return foreground, background, mask2

def enhance_results(foreground, background, mask):
    """Enhance the foreground extraction with edge refinement"""
    # Apply median blur to smooth out noise in the mask
    refined_mask = cv2.medianBlur(mask, 5)
    
    # Optional: Apply morphological operations to further improve the mask
    kernel = np.ones((5,5), np.uint8)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
    
    # Apply the refined mask
    enhanced_foreground = cv2.bitwise_and(foreground, foreground, 
                                          mask=refined_mask)
    
    inv_refined_mask = cv2.bitwise_not(refined_mask)
    enhanced_background = cv2.bitwise_and(background, background,
                                         mask=inv_refined_mask)
    
    return enhanced_foreground, enhanced_background

def save_images(original, foreground, background):
    """Save all three images with timestamp"""
    # Create folder if it doesn't exist
    if not os.path.exists("results"):
        os.makedirs("results")
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save images
    cv2.imwrite(f"results/{timestamp}_original.jpg", original)
    cv2.imwrite(f"results/{timestamp}_foreground.png", foreground)  # PNG for transparency
    cv2.imwrite(f"results/{timestamp}_background.jpg", background)
    
    print(f"Images saved in the 'results' folder with timestamp {timestamp}")

def main():
    print("Starting real-time background removal tool...")
    print("Press ESC to exit")
    
    cap = cv2.VideoCapture(0)  # Open default camera
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Wait for camera to initialize
    for _ in range(10):
        cap.read()
    
    last_capture_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Show live preview
        cv2.imshow('Live Feed (Press ESC to exit)', frame)
        
        current_time = time.time()
        # Check if 10 seconds have passed
        if current_time - last_capture_time >= 10:
            print("\nProcessing frame...")
            
            # Process the current frame
            foreground, background, mask = remove_background(frame)
            enhanced_foreground, enhanced_background = enhance_results(foreground, background, mask)
            
            # Save the images
            save_images(frame, enhanced_foreground, enhanced_background)
            
            # Update the last capture time
            last_capture_time = current_time
            print("Ready for next capture...")
        
        # Check for ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    # Release camera and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()