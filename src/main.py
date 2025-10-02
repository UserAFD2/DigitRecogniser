import cv2
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.models import load_model # type: ignore
import sys

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build the full path to the model
model_path = os.path.join(script_dir, 'digitModel.h5')

if getattr(sys, 'frozen', False):
    # Running in a bundle
    base_path = sys._MEIPASS
else:
    # Running in normal Python
    base_path = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(base_path, "data", "outputs", "digitModel.h5")
# Load the model
model = load_model(model_path)


# Label mapping for the model's output
label_map = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}

# Function to preprocess and segment the drawn image
def segment_image(canvas, padding=20):
    # Preprocess the canvas for contour detection
    _, thresh = cv2.threshold(canvas, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # Find contours of each digit or symbol
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours left to right
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    segments = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 5 and h > 5:  # Filter out noise
            # Add padding, making sure we stay inside the canvas
            x_start = max(x - padding, 0)
            y_start = max(y - padding, 0)
            x_end = min(x + w + padding, canvas.shape[1])
            y_end = min(y + h + padding, canvas.shape[0])

            # Crop the region of interest with padding
            segment = thresh[y_start:y_end, x_start:x_end]

            # Resize to the model's input size (28x28)
            segment = cv2.resize(segment, (28, 28))
            segment = segment.astype('float32') / 255
            segment = segment.reshape(28, 28, 1)  # Model expects this shape
            segments.append(segment)

    return segments, thresh

# Function to predict the segmented digits/symbols
def recognize_segments(segments):
    segments = np.array(segments)
    predictions = model.predict(segments)

    labels = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)  # highest probability for each segment

    recognized = [label_map[label] for label in labels]

    recognized_text = ''.join(recognized)
    confidences_percent = [float(c) * 100 for c in confidences]  # convert each to percentage

    return recognized_text, confidences_percent


# ----------------------------
# Helper functions
# ----------------------------

def create_canvas(width=1500, height=700):
    """Create a white canvas."""
    return np.ones((height, width), dtype=np.uint8) * 255

def create_composite_display(canvas_width, canvas_height):
    """Create composite display area for canvas + predictions."""
    composite_height = canvas_height + 300
    composite_width = max(canvas_width, 800)
    return np.ones((composite_height, composite_width), dtype=np.uint8) * 255, composite_width

def draw_grid(display, canvas_width, canvas_height, grid_size=100):
    """Draw grid lines on the composite display."""
    for i in range(0, canvas_width, grid_size):
        cv2.line(display, (i, 0), (i, canvas_height), (200), 1)
    for j in range(0, canvas_height, grid_size):
        cv2.line(display, (0, j), (canvas_width, j), (200), 1)

def draw_border(display, canvas_width, canvas_height):
    """Draw a border around the canvas."""
    cv2.rectangle(display, (0, 0), (canvas_width - 1, canvas_height - 1), (0), 2)
    
def clear_prediction_area(display, canvas_height, composite_width):
    """Clear prediction and segmented digit areas."""
    display[canvas_height:canvas_height + 100, 0:composite_width] = 255
    display[canvas_height + 100:canvas_height + 300, 0:composite_width] = 255

def show_prediction(display, recognized_text, canvas_height):
    """Show the prediction text below the canvas."""
    cv2.putText(display, f"Prediction: {recognized_text}", (10, canvas_height + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0), 2, cv2.LINE_AA)

def show_segments(display, segments, canvas_height, composite_width, confidences_percent):
    """Display segmented digits below the prediction text."""
    segment_start_y = canvas_height + 100
    x_offset = 10
    for i, segment in enumerate(segments):
        segment_img = (segment.reshape(28, 28) * 255).astype(np.uint8)
        resized_segment = cv2.resize(segment_img, (100, 100))
        y_offset = segment_start_y
        x_position = x_offset + i * 110
        if x_position + 100 > composite_width:
            break
        display[y_offset:y_offset + 100, x_position:x_position + 100] = resized_segment
        cv2.putText(display, f"{confidences_percent[i]:.0f}%", (x_position, y_offset + 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0), 2, cv2.LINE_AA)

def handle_prediction(canvas, composite_display, canvas_height, canvas_width, composite_width):
    """Handle prediction logic when spacebar is pressed."""
    clear_prediction_area(composite_display, canvas_height, composite_width)

    segments, thresh = segment_image(canvas)
    recognized_text, confidences_percent = recognize_segments(segments)

    show_prediction(composite_display, recognized_text, canvas_height)
    show_segments(composite_display, segments, canvas_height, composite_width, confidences_percent)

    cv2.imshow("Digit Recognizer", composite_display)
    cv2.waitKey(2000)

    # Reset canvas in-place
    canvas.fill(255)
    return canvas  # still return it if needed, but it's the same array

def mouse_draw(canvas_ref):
    """Return a mouse callback function that allows drawing on the canvas."""
    state = {"drawing": False, "last_point": None}
    brush_size = 10

    def callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            cv2.circle(canvas_ref, (x, y), brush_size, (0), -1)
            state["last_point"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            if state["last_point"]:
                cv2.line(canvas_ref, state["last_point"], (x, y), (0), brush_size)
            state["last_point"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            state["drawing"] = False
            state["last_point"] = None
    return callback

# ----------------------------
# Main function
# ----------------------------

def draw_and_predict_expression():
    canvas_width, canvas_height = 1500, 700
    canvas = create_canvas(canvas_width, canvas_height)
    composite_display, composite_width = create_composite_display(canvas_width, canvas_height)

    cv2.namedWindow("Digit Recognizer")
    cv2.setMouseCallback("Digit Recognizer", mouse_draw(canvas))

    print("Draw an expression and press SPACE to recognize it. Press C to clear, Q to quit.")

    while True:
        # Update display
        composite_display[:canvas_height, :canvas_width] = canvas
        draw_grid(composite_display, canvas_width, canvas_height)
        draw_border(composite_display, canvas_width, canvas_height)

        cv2.imshow("Digit Recognizer", composite_display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # Predict
            canvas = handle_prediction(canvas, composite_display,
                                       canvas_height, canvas_width, composite_width)

        elif key == ord('c'):  # Clear
            canvas.fill(255)
            composite_display.fill(255)
            print("Canvas cleared!")

        elif key == ord('q'):  # Quit
            print("Quitting without prediction.")
            break

    cv2.destroyAllWindows()
    
# Call this function to start the interactive drawing
draw_and_predict_expression()