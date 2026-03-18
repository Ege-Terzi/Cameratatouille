from ultralytics import YOLO
import cv2
import os
import numpy as np

# Path to the YOLO model file
YOLO_MODEL_PATH = "yolov8x.pt"

# Path to the input video file
VIDEO_PATH = "cooking_video.mp4"

# Directory where cropped bowl images will be saved
SAVE_DIR = "bowl_crops"

# Minimum confidence score for valid detections
CONF_THRESHOLD = 0.5

# YOLO class names that should be treated as bowls
TARGET_BOWL_NAMES = {"bowl"}

# Save one crop every N frames
SAVE_EVERY_N_FRAMES = 10

# Minimum average pixel difference required to consider a crop new enough
MIN_PIXEL_DIFF = 8.0

# Create output directory if it does not already exist
os.makedirs(SAVE_DIR, exist_ok=True)

# Load YOLO model
model = YOLO(YOLO_MODEL_PATH)

# Open input video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

# Playback state
paused = False

# Current frame index
frame_id = 0

# Counter for saved crops (currently not actively used)
saved_count = 0

# Store collected crops in memory until user presses 's'
collected_crops = []

# Keep the last accepted crop in small form to filter very similar crops
last_saved_crop_small = None

def clamp_box(x1, y1, x2, y2, w, h):
    # Clamp bounding box coordinates so they stay inside frame boundaries
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    return x1, y1, x2, y2

def is_different_enough(crop, last_crop_small, threshold=MIN_PIXEL_DIFF):
    # If there is no previous crop, accept the current one
    if last_crop_small is None:
        return True

    # Resize current crop for fast comparison
    small = cv2.resize(crop, (64, 64))

    # Compute mean absolute pixel difference
    diff = np.mean(cv2.absdiff(small, last_crop_small))

    # Accept crop only if difference is above threshold
    return diff >= threshold

while True:
    if not paused:
        # Read next video frame
        ret, frame = cap.read()
        if not ret:
            print("Video ended.")
            break

        # Copy frame for display annotations
        display = frame.copy()

        # Get frame height and width
        h, w = frame.shape[:2]

        # Run YOLO detection on the current frame
        results = model(frame, verbose=False)

        # Iterate through all detection boxes
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            label = model.names[cls_id]

            # Skip detections below confidence threshold
            if conf < CONF_THRESHOLD:
                continue

            # Process only bowl detections
            if label in TARGET_BOWL_NAMES:
                # Get and clamp bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)

                # Draw bounding box around detected bowl
                cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 255), 2)

                # Draw label and confidence score
                cv2.putText(
                    display,
                    f"bowl {conf:.2f}",
                    (x1, max(25, y1)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

                # Collect crop only every N frames
                if frame_id % SAVE_EVERY_N_FRAMES == 0:
                    crop = frame[y1:y2, x1:x2]

                    # Skip invalid or empty crops
                    if crop.size > 0:
                        # Resize crop for similarity comparison
                        small = cv2.resize(crop, (64, 64))

                        # Save crop only if sufficiently different from previous accepted crop
                        if is_different_enough(crop, last_saved_crop_small):
                            collected_crops.append(crop.copy())
                            last_saved_crop_small = small

                            # Show live collected crop count on screen
                            cv2.putText(
                                display,
                                f"collected: {len(collected_crops)}",
                                (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.0,
                                (0, 255, 0),
                                2
                            )

        # Draw keyboard help text
        info = "[p] pause/resume   [s] save and exit   [q] quit"
        cv2.putText(display, info, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw total number of collected crops
        cv2.putText(display, f"Collected crops: {len(collected_crops)}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Show annotated frame
        cv2.imshow("Bowl Crop Collector", display)

        # Move to next frame index
        frame_id += 1

    # Wait for keyboard input
    key = cv2.waitKey(30) & 0xFF

    # Pause or resume playback
    if key == ord('p'):
        paused = not paused
        print("Paused" if paused else "Resumed")

    # Save all collected crops and exit
    elif key == ord('s'):
        print(f"Saving {len(collected_crops)} crops...")
        for i, crop in enumerate(collected_crops):
            path = os.path.join(SAVE_DIR, f"bowl_{i:05d}.jpg")
            cv2.imwrite(path, crop)
        print(f"Saved to folder: {SAVE_DIR}")
        break

    # Exit without saving
    elif key == ord('q'):
        print("Exited without saving.")
        break

# Release video resource and close display windows
cap.release()
cv2.destroyAllWindows()