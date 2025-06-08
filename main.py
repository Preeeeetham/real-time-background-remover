import cv2
import numpy as np
import os
import argparse
from datetime import datetime
import time

def parse_arguments():
    parser = argparse.ArgumentParser(description="Real-time background removal tool using OpenCV's GrabCut algorithm.")
    parser.add_argument("--interval", type=int, default=10, help="Interval (seconds) between frame processing.")
    parser.add_argument("--scale", type=float, default=0.5, help="Scale factor for frame resolution (e.g., 0.5 for half resolution).")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save output images.")
    return parser.parse_args()

def capture_image(cap):
    print("Starting camera... Press SPACE to capture or ESC to quit")
    
    if not cap.isOpened():
        print("Could not open camera.")
        return None
    for _ in range(10):
        cap.read()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        cv2.imshow('Press SPACE to capture, ESC to quit', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            return None
        elif key == 32:
            return frame
    return None

def resize_frame(frame, scale):
    if scale != 1.0:
        h, w = frame.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame

def remove_background(image):
    start_time = time.time()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    h, w = image.shape[:2]
    margin = min(h, w) // 8
    rect = (margin, margin, w - 2*margin, h - 2*margin)
    print(f"Processing image ({w}x{h}) with GrabCut...")
    cv2.grabCut(image_rgb, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    foreground = image.copy()
    foreground = cv2.bitwise_and(foreground, foreground, mask=mask2)
    background = image.copy()
    inv_mask = cv2.bitwise_not(mask2)
    background = cv2.bitwise_and(background, background, mask=inv_mask)
    print(f"GrabCut processing took {time.time() - start_time:.2f} seconds")
    return foreground, background, mask2

def enhance_results(foreground, background, mask):
    refined_mask = cv2.medianBlur(mask, 5)
    kernel = np.ones((5,5), np.uint8)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
    enhanced_foreground = cv2.bitwise_and(foreground, foreground, mask=refined_mask)
    inv_refined_mask = cv2.bitwise_not(refined_mask)
    enhanced_background = cv2.bitwise_and(background, background, mask=inv_refined_mask)
    return enhanced_foreground, enhanced_background

def save_images(original, foreground, background, output_dir):
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(os.path.join(output_dir, f"{timestamp}_original.jpg"), original)
        cv2.imwrite(os.path.join(output_dir, f"{timestamp}_foreground.png"), foreground)
        cv2.imwrite(os.path.join(output_dir, f"{timestamp}_background.jpg"), background)
        print(f"Images saved in '{output_dir}' with timestamp {timestamp}")
    except Exception as e:
        print(f"Error saving images: {e}")

def main():
    args = parse_arguments()
    print(f"Starting real-time background removal tool (interval: {args.interval}s, scale: {args.scale})...")
    print("Press ESC to exit, 'P' to toggle processing")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    for _ in range(10):
        cap.read()
    
    last_capture_time = time.time()
    processing_enabled = True
    last_foreground, last_background = None, None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        display_frame = resize_frame(frame.copy(), args.scale)
        
        if last_foreground is not None and last_background is not None:
            last_foreground_resized = resize_frame(last_foreground, args.scale)
            last_background_resized = resize_frame(last_background, args.scale)
            combined = np.hstack((display_frame, last_foreground_resized, last_background_resized))
            cv2.imshow('Live Feed | Foreground | Background (ESC to exit, P to toggle)', combined)
        else:
            cv2.imshow('Live Feed | Foreground | Background (ESC to exit, P to toggle)', display_frame)

        current_time = time.time()
        if processing_enabled and (current_time - last_capture_time >= args.interval):
            print("\nProcessing frame...")
            foreground, background, mask = remove_background(frame)
            enhanced_foreground, enhanced_background = enhance_results(foreground, background, mask)
            save_images(frame, enhanced_foreground, enhanced_background, args.output_dir)
            last_foreground, last_background = enhanced_foreground, enhanced_background
            last_capture_time = current_time
            print("Ready for next capture...")

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('p'):
            processing_enabled = not processing_enabled
            print(f"Processing {'enabled' if processing_enabled else 'disabled'}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
