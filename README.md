# Fire Detection System

A real-time fire detection web application built with **Streamlit** and a **YOLO** object detection model. It supports detecting fire in uploaded images, uploaded videos, and a live camera feed.

## Features

- **Image Upload** — Upload a single image and run fire detection on it, with a side-by-side view of the original and annotated result.
- **Video Upload** — Upload a video file, process it frame-by-frame (every 10th frame for speed), and view live progress plus a running detection count.
- **Live Camera** — Run detection directly on a connected webcam feed, with adjustable camera index and detection sensitivity, and live "fire alert" status updates.
- **Adjustable Confidence Threshold** — Sidebar slider to tune detection sensitivity.
- **Detection Statistics** — Running total of detections across the session, with a reset button.

## How It Works

1. A YOLO model (`fire.pt`) is loaded once and cached via `@st.cache_resource`.
2. Each frame (from an image, video, or camera) is passed through the model.
3. Detections above the confidence threshold (default 50%) are drawn as bounding boxes with confidence labels using `cvzone`.
4. Results are displayed back to the user in real time, along with alerts when fire is detected.

## Requirements

```
streamlit
opencv-python
numpy
Pillow
ultralytics
cvzone
```

You will also need a trained YOLO fire-detection weights file named **`fire.pt`** in the project directory (not included — must be supplied separately, e.g. trained on a fire-detection dataset via Ultralytics YOLO).

## Installation

```bash
pip install streamlit opencv-python numpy Pillow ultralytics cvzone
```

Place your `fire.pt` model weights in the same directory as `fire_detection_webapp.py`.

## Usage

Run the app with:

```bash
streamlit run fire_detection_webapp.py
```

Then open the local URL Streamlit provides (typically `http://localhost:8501`) in your browser.

### Tabs

| Tab | Description |
|---|---|
| Image Upload | Upload a `.jpg`, `.jpeg`, `.png`, or `.bmp` image for detection |
| Video Upload | Upload a `.mp4`, `.avi`, `.mov`, or `.mkv` video for frame-by-frame detection |
| Live Camera | Select a camera index and run real-time detection on the webcam feed |

## Known Limitations / Notes

- The live camera loop currently runs until `frame_counter > 1000` — there's no explicit **Stop** button, so long-running sessions will keep the camera open until that frame limit is reached.
- Video processing samples every 10th frame, so very short fire events between sampled frames could be missed.
- Camera access depends on OpenCV having permission to your system's camera devices, not just browser permissions — the in-app troubleshooting tips reference browser settings, but the actual camera handle is opened server-side via `cv2.VideoCapture`.
- No persistence of detection history between sessions — the detection counter resets on page reload unless manually tracked elsewhere.

## Project Structure

```
.
├── fire_detection_webapp.py   # Main Streamlit application
├── fire.pt                    # YOLO fire-detection model weights (not included)
└── README.md
```

