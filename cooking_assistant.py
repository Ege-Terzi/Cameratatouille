from ultralytics import YOLO
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from collections import Counter, deque
import json
from pathlib import Path

# Import UI-related helper functions and layout constants
from ui import (
    choose_recipe_from_terminal,
    build_info_lines,
    build_step_line,
    get_help_text,
    get_step_color,
    get_step_start_y,
    get_max_visible_steps,
    get_visible_step_range,
    LEFT_MARGIN,
    TOP_MARGIN,
    INFO_LINE_HEIGHT,
    STEP_LINE_HEIGHT,
    HELP_TEXT_Y_OFFSET,
    OVERLAY_TOP,
    OVERLAY_LEFT,
    OVERLAY_RIGHT,
    OVERLAY_BOTTOM,
)

# =========================================================
# CONFIG
# =========================================================

# Path to the YOLO object detection model
YOLO_MODEL_PATH = "yolov8x.pt"

# Input video file path
VIDEO_PATH = "cooking_video.mp4"

# Output video file path
OUTPUT_PATH = "output_assistant.mp4"

# Enable or disable the custom bowl content classifier
USE_BOWL_CLASSIFIER = True

# Path to the trained bowl classifier weights
BOWL_CLASSIFIER_WEIGHTS = "bowl_classifier.pth"

# Minimum confidence threshold for YOLO detections
CONF_THRESHOLD = 0.45

# Number of repeated detections required before considering an ingredient stable
STABLE_COUNT_THRESHOLD = 5

# Number of recent object detections to keep in memory
RECENT_DETECTIONS_WINDOW = 60

# Number of recent bowl states to keep for smoothing predictions
RECENT_BOWL_STATES_WINDOW = 30

# IMPORTANT: This order must match the order used during training
# {'beaten_egg_bowl': 0, 'empty_bowl': 1}
BOWL_CONTENT_CLASSES = [
    "beaten_egg_bowl",
    "empty_bowl"
]

# YOLO class names that should be treated as bowls
TARGET_BOWL_NAMES = {"bowl"}

# Map raw YOLO labels to simplified or custom labels
LABEL_MAP = {
    "banana": "banana",
    "apple": "apple",
    "orange": "orange",
    "bowl": "bowl",
    "egg": "egg",
    "tomato": "tomato",
    "broccoli": "broccoli",
    "carrot": "carrot",
    "cup": "cup",
    "bottle": "bottle",
    "sandwich": "sandwich",
    "pizza": "pizza",
    "hot dog": "sausage",
    "donut": "donut",
    "cake": "cake",
}

# Convert detected bowl content state into implied ingredients
BOWL_STATE_TO_INGREDIENTS = {
    "beaten_egg_bowl": ["egg"],
    "empty_bowl": []
}

# =========================================================
# RECIPES
# =========================================================

# Get the directory of the current Python file
BASE_DIR = Path(__file__).resolve().parent

# Path to the recipes JSON file
RECIPES_PATH = BASE_DIR / "recipes.json"


def load_recipes(json_path=RECIPES_PATH):
    # Load and return recipe data from a JSON file
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Load all recipes at startup
RECIPES = load_recipes()

# =========================================================
# IMAGE ENHANCEMENT
# =========================================================

# Create CLAHE object for local contrast enhancement
clahe = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8))

def enhance_frame(frame):
    # Convert the frame to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # Split LAB channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the lightness channel
    l_clahe = clahe.apply(l)

    # Merge enhanced lightness back with original color channels
    lab_clahe = cv2.merge((l_clahe, a, b))

    # Convert back to BGR color space
    enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return enhanced

def unsharp_mask(image):
    # Blur the image slightly
    blur = cv2.GaussianBlur(image, (5, 5), 0)

    # Sharpen the image by combining original and blurred versions
    sharp = cv2.addWeighted(image, 1.35, blur, -0.35, 0)
    return sharp

# =========================================================
# BOWL CLASSIFIER
# =========================================================

def build_bowl_classifier(num_classes):
    # Load a pretrained MobileNetV3 Small model
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

    # Get the number of input features of the final classifier layer
    in_features = model.classifier[3].in_features

    # Replace the final classification layer with a new one for our custom classes
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model

# Define image preprocessing pipeline for bowl classification
classifier_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to model input size
    transforms.ToTensor(),          # Convert PIL image to tensor
    transforms.Normalize(           # Normalize using ImageNet mean and std
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_bowl_classifier(device):
    # Build the classifier architecture
    model = build_bowl_classifier(len(BOWL_CONTENT_CLASSES))

    # Load trained weights
    state = torch.load(BOWL_CLASSIFIER_WEIGHTS, map_location=device)

    # Apply weights to the model
    model.load_state_dict(state)

    # Move model to the selected device
    model.to(device)

    # Switch model to evaluation mode
    model.eval()
    return model

@torch.no_grad()
def classify_bowl_content(crop_bgr, classifier_model, device):
    # Return unknown if the bowl crop is empty or invalid
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown", 0.0

    # Convert crop from BGR to RGB
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    # Convert NumPy array to PIL image
    pil_img = Image.fromarray(crop_rgb)

    # Apply preprocessing and add batch dimension
    x = classifier_transform(pil_img).unsqueeze(0).to(device)

    # Run inference
    logits = classifier_model(x)

    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=1)

    # Get the most likely class and its confidence
    conf, pred = torch.max(probs, dim=1)

    # Map predicted class index to class label
    label = BOWL_CONTENT_CLASSES[pred.item()]
    return label, float(conf.item())

# =========================================================
# HELPERS
# =========================================================

def clamp_box(x1, y1, x2, y2, w, h):
    # Clamp bounding box coordinates so they stay inside image boundaries
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    return x1, y1, x2, y2

def draw_text_lines(img, lines, x, y, line_height=26, color=(255, 255, 255), scale=0.65, thickness=2):
    # Draw multiple lines of text on an image
    yy = y
    for line in lines:
        cv2.putText(img, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
        yy += line_height

def get_missing_ingredients(recipe, stable_ingredients):
    # Return ingredients required by the recipe but not currently detected
    stable = set(stable_ingredients)
    return [ing for ing in recipe["ingredients"] if ing not in stable]

def step_is_complete(step_rule, stable_ingredients, stable_bowl_state):
    # Return False if there is no completion rule
    if not step_rule:
        return False

    # Assume both conditions are satisfied unless proven otherwise
    ingredients_ok = True
    bowl_ok = True

    # Check if all required ingredients are present
    if "ingredients_present" in step_rule:
        needed = step_rule["ingredients_present"]
        ingredients_ok = all(x in stable_ingredients for x in needed)

    # Check if bowl state is one of the allowed states
    if "bowl_state_in" in step_rule:
        allowed_states = step_rule["bowl_state_in"]
        bowl_ok = stable_bowl_state in allowed_states

    # Step is complete only if all required conditions are satisfied
    return ingredients_ok and bowl_ok

def detect_stable_ingredients(results, model_names, recent_detections):
    # Temporary list of items detected in the current frame
    frame_items = []

    # Iterate over YOLO detection boxes
    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())

        # Ignore low-confidence detections
        if conf < CONF_THRESHOLD:
            continue

        # Get class name and map it to custom label if needed
        raw_label = model_names[cls_id]
        mapped = LABEL_MAP.get(raw_label, raw_label)
        frame_items.append(mapped)

    # Add current frame detections to rolling history
    recent_detections.extend(frame_items)

    # Count occurrences in the detection history
    counts = Counter(recent_detections)

    # Keep only ingredients seen enough times to be considered stable
    return [k for k, v in counts.items() if v >= STABLE_COUNT_THRESHOLD]

def enrich_ingredients_with_bowl_state(stable_ingredients, stable_bowl_state):
    # Start with currently stable ingredients
    enriched = set(stable_ingredients)

    # Add implied ingredients based on bowl content state
    if stable_bowl_state in BOWL_STATE_TO_INGREDIENTS:
        for item in BOWL_STATE_TO_INGREDIENTS[stable_bowl_state]:
            enriched.add(item)

    return list(enriched)

def get_stable_bowl_state(results, frame_for_crop, yolo_model, bowl_classifier, device, recent_bowl_states):
    # Skip bowl state classification if classifier is disabled or unavailable
    if not USE_BOWL_CLASSIFIER or bowl_classifier is None:
        return "unknown", 0.0, None

    h, w = frame_for_crop.shape[:2]
    best_candidate = None

    # Search for the best bowl detection in the frame
    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        label = yolo_model.names[cls_id]

        # Ignore low-confidence detections or non-bowl objects
        if conf < CONF_THRESHOLD or label not in TARGET_BOWL_NAMES:
            continue

        # Extract and clamp bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)

        # Compute area to choose the largest bowl candidate
        area = max(0, x2 - x1) * max(0, y2 - y1)

        # Keep the largest detected bowl
        if best_candidate is None or area > best_candidate["area"]:
            best_candidate = {
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "area": area
            }

    # Return unknown if no bowl was found
    if best_candidate is None:
        return "unknown", 0.0, None

    # Crop the bowl region from the frame
    bowl_crop = frame_for_crop[
        best_candidate["y1"]:best_candidate["y2"],
        best_candidate["x1"]:best_candidate["x2"]
    ]

    # Classify the content of the bowl
    bowl_state, bowl_score = classify_bowl_content(bowl_crop, bowl_classifier, device)

    # Store only sufficiently confident bowl predictions
    if bowl_score > 0.50:
        recent_bowl_states.append(bowl_state)

    # Use the most common recent bowl state for stability
    stable_state = bowl_state
    if len(recent_bowl_states) > 0:
        stable_state = Counter(recent_bowl_states).most_common(1)[0][0]

    return stable_state, bowl_score, best_candidate

# =========================================================
# MAIN
# =========================================================

def main():
    # Let the user select a recipe from the terminal
    selected_recipe_idx = choose_recipe_from_terminal(RECIPES)
    selected_recipe = RECIPES[selected_recipe_idx]

    print(f"\nSelected recipe: {selected_recipe['name']}")
    print("Video is starting...\n")

    # Choose GPU if available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Load the YOLO model
    yolo_model = YOLO(YOLO_MODEL_PATH)
    yolo_model.to(device)

    # Try to load the bowl classifier if enabled
    bowl_classifier = None
    if USE_BOWL_CLASSIFIER:
        try:
            bowl_classifier = load_bowl_classifier(device)
            print("Bowl classifier loaded.")
        except Exception as e:
            print("Bowl classifier could not be loaded:", e)
            print("Continuing without bowl classifier.")
            bowl_classifier = None

    # Open the input video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    # Read basic video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    # Rolling buffers for stabilizing detections
    recent_detections = deque(maxlen=RECENT_DETECTIONS_WINDOW)
    recent_bowl_states = deque(maxlen=RECENT_BOWL_STATES_WINDOW)

    # State variables for recipe progression and controls
    current_step_idx = 0
    completed_steps = set()
    auto_mode = True
    paused = False

    # Main video processing loop
    while True:
        if not paused:
            # Read next frame
            ret, frame = cap.read()
            if not ret:
                print("Video ended.")
                break

            # Improve frame quality before detection
            enhanced = enhance_frame(frame)
            sharpened = unsharp_mask(enhanced)

            # Run YOLO object detection
            results = yolo_model(sharpened, verbose=False)

            # Create YOLO-annotated frame
            annotated = results[0].plot()

            # Detect stable ingredients over recent frames
            stable_ingredients = detect_stable_ingredients(results, yolo_model.names, recent_detections)

            # Detect and stabilize bowl state
            stable_bowl_state, bowl_score, bowl_box = get_stable_bowl_state(
                results, sharpened, yolo_model, bowl_classifier, device, recent_bowl_states
            )

            # Add implied ingredients from bowl content state
            stable_ingredients = enrich_ingredients_with_bowl_state(stable_ingredients, stable_bowl_state)

            # Draw bowl rectangle and label if a bowl was found
            if bowl_box is not None:
                cv2.rectangle(
                    annotated,
                    (bowl_box["x1"], bowl_box["y1"]),
                    (bowl_box["x2"], bowl_box["y2"]),
                    (255, 0, 255),
                    2
                )
                cv2.putText(
                    annotated,
                    f"{stable_bowl_state} {bowl_score:.2f}",
                    (bowl_box["x1"], max(25, bowl_box["y1"])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

            # Auto-advance to the next recipe step if current step is complete
            if current_step_idx < len(selected_recipe["steps"]):
                rule = selected_recipe["steps"][current_step_idx].get("complete_if", {})
                if auto_mode and step_is_complete(rule, stable_ingredients, stable_bowl_state):
                    completed_steps.add(current_step_idx)
                    if current_step_idx < len(selected_recipe["steps"]) - 1:
                        current_step_idx += 1

            # Create a semi-transparent overlay panel for text information
            overlay = annotated.copy()
            cv2.rectangle(overlay, (OVERLAY_LEFT, OVERLAY_TOP), (OVERLAY_RIGHT, OVERLAY_BOTTOM), (0, 0, 0), -1)
            final_frame = cv2.addWeighted(overlay, 0.42, annotated, 0.58, 0)

            # Compute missing ingredients
            missing = get_missing_ingredients(selected_recipe, stable_ingredients)

            # Build and draw status/info lines
            info_lines = build_info_lines(
                selected_recipe,
                stable_ingredients,
                stable_bowl_state,
                missing,
                current_step_idx,
                auto_mode,
            )
            draw_text_lines(final_frame, info_lines, LEFT_MARGIN, TOP_MARGIN, line_height=INFO_LINE_HEIGHT)

            # Determine which recipe steps fit on screen
            step_start_y = get_step_start_y(len(info_lines))
            max_visible_steps = get_max_visible_steps(height, step_start_y)
            start_idx, end_idx = get_visible_step_range(
                len(selected_recipe["steps"]),
                current_step_idx,
                max_visible_steps,
            )

            # Draw visible recipe steps
            for visible_idx, i in enumerate(range(start_idx, end_idx)):
                step = selected_recipe["steps"][i]
                color = get_step_color(i, completed_steps, current_step_idx)

                cv2.putText(
                    final_frame,
                    build_step_line(i, step, completed_steps, current_step_idx),
                    (LEFT_MARGIN, step_start_y + visible_idx * STEP_LINE_HEIGHT),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

            # Draw keyboard help text at the bottom
            help_text = get_help_text()
            cv2.putText(final_frame, help_text, (20, height - HELP_TEXT_Y_OFFSET), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # Show frame in a window
            cv2.imshow("Cooking Assistant", final_frame)

            # Write frame to output video
            out.write(final_frame)

        # Handle keyboard input
        key = cv2.waitKey(1 if not paused else 30) & 0xFF

        # Quit
        if key == ord("q"):
            break

        # Pause / resume playback
        elif key == ord(" "):
            paused = not paused

        # Toggle auto mode
        elif key == ord("a"):
            auto_mode = not auto_mode

        # Move to next step manually
        elif key == ord("n"):
            if current_step_idx < len(selected_recipe["steps"]) - 1:
                current_step_idx += 1

        # Move to previous step manually
        elif key == ord("p"):
            if current_step_idx > 0:
                current_step_idx -= 1

        # Mark current step as completed / uncompleted
        elif key == ord("c"):
            if current_step_idx in completed_steps:
                completed_steps.remove(current_step_idx)
            else:
                completed_steps.add(current_step_idx)
                if current_step_idx < len(selected_recipe["steps"]) - 1:
                    current_step_idx += 1

    # Release video resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Run the program only if this file is executed directly
if __name__ == "__main__":
    main()