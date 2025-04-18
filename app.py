import cv2
import numpy as np
import os
from datetime import datetime
import time

def capture_image():
    """Capture an image from the camera"""
    print("Starting camera... Press SPACE to capture or ESC to quit")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None
    for _ in range(10):
        cap.read()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow('Press SPACE to capture, ESC to quit', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27: 
            break
        elif key == 32:
            cap.release()
            cv2.destroyAllWindows()
            return frame
    cap.release()
    cv2.destroyAllWindows()
    return None

def remove_background(image):
    """Remove background using GrabCut algorithm with automatic initialization"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    h, w = image.shape[:2]
    margin = min(h, w) // 8 
    rect = (margin, margin, w - 2*margin, h - 2*margin)
    print("Processing image with GrabCut (this may take a moment)...")
    cv2.grabCut(image_rgb, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    foreground = image.copy()
    foreground = cv2.bitwise_and(foreground, foreground, mask=mask2)
    background = image.copy()
    inv_mask = cv2.bitwise_not(mask2)
    background = cv2.bitwise_and(background, background, mask=inv_mask)
    
    return foreground, background, mask2

def enhance_results(foreground, background, mask):
    """Enhance the foreground extraction with edge refinement"""
    refined_mask = cv2.medianBlur(mask, 5)
    kernel = np.ones((5,5), np.uint8)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
    enhanced_foreground = cv2.bitwise_and(foreground, foreground, 
                                          mask=refined_mask)
    inv_refined_mask = cv2.bitwise_not(refined_mask)
    enhanced_background = cv2.bitwise_and(background, background,
                                         mask=inv_refined_mask)
    return enhanced_foreground, enhanced_background
    
def save_images(original, foreground, background):
    if not os.path.exists("results"):
        os.makedirs("results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(f"results/{timestamp}_original.jpg", original)
    cv2.imwrite(f"results/{timestamp}_foreground.png", foreground)
    cv2.imwrite(f"results/{timestamp}_background.jpg", background)
    print(f"Images saved in the 'results' folder with timestamp {timestamp}")

def main():
    print("Starting real-time background removal tool...")
    print("Press ESC to exit")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    for _ in range(10):
        cap.read()
    last_capture_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow('Live Feed (Press ESC to exit)', frame)
        
        current_time = time.time()
        if current_time - last_capture_time >= 10:
            print("\nProcessing frame...")
            foreground, background, mask = remove_background(frame)
            enhanced_foreground, enhanced_background = enhance_results(foreground, background, mask)
            save_images(frame, enhanced_foreground, enhanced_background)
            last_capture_time = current_time
            print("Ready for next capture...")
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
