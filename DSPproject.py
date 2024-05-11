import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Function to process frame
def process_frame(frame, blur_intensity=10):
    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred_frame = custom_blur(frame, blur_intensity)
    
    # Apply edge detection
    edges = cv2.Canny(frame, 50, 70)
    
    # Apply histogram equalization
    equalized_frame = cv2.equalizeHist(gray_frame)
    
    # Apply coloring
    colored_frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
    
    # Apply invert colors
    inverted_frame = cv2.bitwise_not(frame)
    
    return gray_frame, blurred_frame, edges, equalized_frame, colored_frame, inverted_frame

#blurred one
def custom_blur(image, blur_intensity=10):
    # Define a blur kernel
    filter_size = blur_intensity * 2 + 1
    blur_kernel = np.ones((filter_size, filter_size), np.float32) / (filter_size ** 2)
    
    # Apply the blur kernel
    blurred_image = cv2.filter2D(image, -1, blur_kernel)
    
    return blurred_image

# Load video
video_path = 'D:/level-3-cis/DSP/CIS.mp4'
cap = cv2.VideoCapture(video_path)

# Create directory to save processed frames
processed_frames_dir = r'D:\level-3-cis\DSP\processed_frames'
os.makedirs(processed_frames_dir, exist_ok=True)

frame_count = 0
max_frame_index = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1 #all frames in video -1 => index 0
while frame_count < 10:
    # Choose a random frame index
    random_frame_index = np.random.randint(0, max_frame_index)
    
    # Set the frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_index) # go direct to the frame 
    
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Save original frame
    original_frame_path = os.path.join(processed_frames_dir, f'frame_{frame_count}_original.jpg')
    cv2.imwrite(original_frame_path, frame)
    
    # Process frame
    gray_frame, blurred_frame, edges, equalized_frame, colored_frame, inverted_frame = process_frame(frame, blur_intensity=20)
    
    # Save processed frames
    cv2.imwrite(os.path.join(processed_frames_dir, f'frame_{frame_count}_gray.jpg'), gray_frame)
    cv2.imwrite(os.path.join(processed_frames_dir, f'frame_{frame_count}_colored.jpg'), colored_frame)
    cv2.imwrite(os.path.join(processed_frames_dir, f'frame_{frame_count}_blurred.jpg'), blurred_frame)
    cv2.imwrite(os.path.join(processed_frames_dir, f'frame_{frame_count}_edges.jpg'), edges)
    cv2.imwrite(os.path.join(processed_frames_dir, f'frame_{frame_count}_equalized.jpg'), equalized_frame)
    cv2.imwrite(os.path.join(processed_frames_dir, f'frame_{frame_count}_inverted.jpg'), inverted_frame)
    
    frame_count += 1

cap.release()

# Display original and processed frames for a few samples
sample_frame_indices = np.random.choice(frame_count, min(1, frame_count), replace=False) #choose at least one frame
sample_frames = [] # original & processed
for frame_index in sample_frame_indices: #for the random frame 
    original_frame_path = os.path.join(processed_frames_dir, f'frame_{frame_index}_original.jpg')
    original_frame = cv2.imread(original_frame_path)
    processed_frames = [original_frame] 
    for i in range(6):  # for processes
        processed_frame_path = os.path.join(processed_frames_dir, f'frame_{frame_index}_{["gray", "blurred", "edges", "equalized", "colored", "inverted"][i]}.jpg')
        if os.path.exists(processed_frame_path):  # Check if processed frame exists
            processed_frame = cv2.imread(processed_frame_path)
            processed_frames.append(processed_frame)
    sample_frames.append(processed_frames)

plt.figure(figsize=(12, 8))  # Adjust the figure size here
for i, frame_index in enumerate(sample_frame_indices):
    titles = ['Original', 'Grayscale', 'Blurred', 'Edges', 'Equalized', 'Colored', 'Inverted']
    for j in range(7):
        plt.subplot(4, 7, i * 7 + j + 1)
        if j == 0:
            plt.imshow(cv2.cvtColor(sample_frames[i][j], cv2.COLOR_BGR2RGB), aspect='auto')  # Adjust aspect ratio to make images larger
            plt.title(f'Frame Index: {frame_index}')
        elif j < len(sample_frames[i]):  # Check if processed frame exists
            plt.imshow(cv2.cvtColor(sample_frames[i][j], cv2.COLOR_BGR2RGB), aspect='auto')  # Adjust aspect ratio to make images larger
            plt.title(titles[j])
        else:
            plt.axis('off')
        plt.axis('off')

plt.tight_layout()
plt.show()

