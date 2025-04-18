# real-time-background-removerReal-Time Background Removal Tool
This Python script provides a real-time background removal tool using OpenCV's GrabCut algorithm. It captures video from a webcam, processes frames periodically to separate foreground and background, and saves the results with timestamps.
Features

Captures images or video from a webcam
Automatically removes background using GrabCut algorithm
Enhances results with edge refinement
Saves original, foreground, and background images with timestamps
Processes frames every 10 seconds in real-time mode
Interactive interface with live feed preview

Requirements

Python 3.x
OpenCV (opencv-python)
NumPy

Install the required packages using:
pip install opencv-python numpy

Usage

Clone the repository:

git clone <repository-url>
cd <repository-directory>


Run the script:

python background_removal.py


Interaction:


Press ESC to exit the program
In capture mode (if used), press SPACE to capture an image or ESC to quit
The script automatically processes frames every 10 seconds in real-time mode


Output:


Results are saved in the results/ folder with timestamps:
<timestamp>_original.jpg: Original captured frame
<timestamp>_foreground.png: Foreground with background removed
<timestamp>_background.jpg: Background only



How It Works

Capture: Uses OpenCV to capture video frames from the webcam.
Background Removal: Applies the GrabCut algorithm with automatic rectangle initialization to separate foreground and background.
Enhancement: Refines the segmentation using median blur and morphological operations for smoother edges.
Saving: Saves the original frame, extracted foreground, and background as separate images.

File Structure

background_removal.py: Main script containing all functionality
results/: Directory where output images are saved (created automatically)

Notes

Ensure your webcam is connected and accessible.
The GrabCut algorithm may take a moment to process each frame.
Results depend on lighting conditions and contrast between foreground and background.
The script processes frames every 10 seconds to balance performance and usability.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Contributing
Contributions are welcome! Please submit a pull request or open an issue for suggestions or bug reports.
