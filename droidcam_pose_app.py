"""
Installation:
    python -m pip install -r requirements.txt

Recommended Python:
    Python 3.10+ (Tested on 3.13 with MediaPipe Tasks API)

Run examples:
    python droidcam_pose_app.py --source 0
"""

from __future__ import annotations

import argparse
import time
import os
from typing import Dict, Optional, Tuple, Union, List

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    from tensorflow import keras
    from tensorflow.keras import layers
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. CNN motion classification will be disabled.")

# --- Constants & Configuration ---
MODEL_PATH = 'pose_landmarker_lite.task'
CNN_MODEL_PATH = 'motion_classifier_model.h5'

# HOG descriptor parameters
HOG_WIN_SIZE = (64, 128)
HOG_BLOCK_SIZE = (16, 16)
HOG_BLOCK_STRIDE = (8, 8)
HOG_CELL_SIZE = (8, 8)
HOG_NBINS = 9

# Motion classification classes
MOTION_CLASSES = ['standing', 'walking', 'running']

# Standard MediaPipe Pose Landmark Indices
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
JOINT_INDICES = {
    "head": 0,  # Nose
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

SKELETON_EDGES = [
    ("head", "left_shoulder"),
    ("head", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

def load_landmarker(model_path: str) -> vision.PoseLandmarker:
    """Initialize the MediaPipe Pose Landmarker."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file '{model_path}' not found.\n"
            "Please download it using:\n"
            "python -c \"import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task', 'pose_landmarker_lite.task')\""
        )

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)

def preprocess_frame(frame: np.ndarray, blur_kernel: Tuple[int, int] = (3, 3)) -> np.ndarray:
    """Apply Gaussian filtering to reduce frame noise."""
    return cv2.GaussianBlur(frame, blur_kernel, 0)

def detect_humans_hog(frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect humans using HOG (Histogram of Oriented Gradients) descriptor."""
    # Initialize HOG descriptor
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # Detect humans in the frame
    boxes, weights = hog.detectMultiScale(
        frame,
        winStride=(8, 8),
        padding=(32, 32),
        scale=1.05
    )
    
    # Return bounding boxes as (x, y, width, height)
    return boxes.tolist() if len(boxes) > 0 else []

def detect_harris_corners(
    frame: np.ndarray,
    max_corners: int = 120,
    quality_level: float = 0.01,
    min_distance: int = 10,
    block_size: int = 3,
    k: float = 0.04
) -> np.ndarray:
    """Detect corners using Harris corner detection algorithm."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    
    # Apply Harris corner detection
    harris_response = cv2.cornerHarris(
        gray,
        blockSize=block_size,
        ksize=3,
        k=k
    )
    
    # Normalize and threshold
    harris_response = cv2.dilate(harris_response, None)
    threshold = quality_level * harris_response.max()
    
    # Get corner coordinates
    corner_coords = np.argwhere(harris_response > threshold)
    
    if len(corner_coords) == 0:
        return np.array([]).reshape(0, 1, 2)
    
    # Convert to float32 format expected by optical flow
    corners = corner_coords[:, [1, 0]].astype(np.float32).reshape(-1, 1, 2)
    
    # Filter by minimum distance
    if len(corners) > max_corners:
        # Simple distance-based filtering
        filtered_corners = []
        for corner in corners:
            x, y = corner[0]
            too_close = False
            for fc in filtered_corners:
                fx, fy = fc[0]
                if np.sqrt((x - fx)**2 + (y - fy)**2) < min_distance:
                    too_close = True
                    break
            if not too_close:
                filtered_corners.append(corner)
                if len(filtered_corners) >= max_corners:
                    break
        corners = np.array(filtered_corners).reshape(-1, 1, 2)
    
    return corners

def create_motion_classifier_cnn(input_shape: Tuple[int, int, int] = (64, 64, 3), num_classes: int = 3) -> keras.Model:
    """Create a CNN model for motion pattern classification."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_or_create_motion_classifier() -> Optional[keras.Model]:
    """Load existing CNN model or create a new one."""
    if not TF_AVAILABLE:
        return None
    
    if os.path.exists(CNN_MODEL_PATH):
        try:
            model = keras.models.load_model(CNN_MODEL_PATH)
            print(f"Loaded motion classifier from {CNN_MODEL_PATH}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}. Creating new model.")
    
    # Create a new model (untrained, will use rule-based classification as fallback)
    model = create_motion_classifier_cnn()
    print("Created new motion classifier model (untrained). Using rule-based classification.")
    return model

def classify_motion_rule_based(
    keypoints: Dict[str, Tuple[int, int]],
    prev_keypoints: Optional[Dict[str, Tuple[int, int]]],
    frame_count: int
) -> str:
    """Classify motion pattern using rule-based approach based on keypoint movement."""
    if not keypoints or not prev_keypoints:
        return 'standing'
    
    # Calculate movement of key joints
    movement_sum = 0.0
    joint_count = 0
    
    for joint_name in ['left_ankle', 'right_ankle', 'left_knee', 'right_knee', 'left_hip', 'right_hip']:
        if joint_name in keypoints and joint_name in prev_keypoints:
            x1, y1 = keypoints[joint_name]
            x2, y2 = prev_keypoints[joint_name]
            movement = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            movement_sum += movement
            joint_count += 1
    
    if joint_count == 0:
        return 'standing'
    
    avg_movement = movement_sum / joint_count
    
    # Thresholds (pixels per frame)
    if avg_movement < 2.0:
        return 'standing'
    elif avg_movement < 8.0:
        return 'walking'
    else:
        return 'running'

def classify_motion_cnn(
    model: keras.Model,
    frame: np.ndarray,
    keypoints: Dict[str, Tuple[int, int]]
) -> str:
    """Classify motion pattern using CNN model."""
    if not keypoints or len(keypoints) < 4:
        return 'standing'
    
    # Extract region of interest around detected person
    if 'left_hip' in keypoints and 'right_hip' in keypoints:
        # Use hip as center
        center_x = (keypoints['left_hip'][0] + keypoints['right_hip'][0]) // 2
        center_y = (keypoints['left_hip'][1] + keypoints['right_hip'][1]) // 2
    elif 'head' in keypoints:
        center_x, center_y = keypoints['head']
    else:
        return 'standing'
    
    # Extract 64x64 region
    h, w = frame.shape[:2]
    x1 = max(0, center_x - 32)
    y1 = max(0, center_y - 32)
    x2 = min(w, center_x + 32)
    y2 = min(h, center_y + 32)
    
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return 'standing'
    
    # Resize to model input size
    roi_resized = cv2.resize(roi, (64, 64))
    roi_normalized = roi_resized.astype(np.float32) / 255.0
    roi_batch = np.expand_dims(roi_normalized, axis=0)
    
    try:
        predictions = model.predict(roi_batch, verbose=0)
        class_idx = np.argmax(predictions[0])
        return MOTION_CLASSES[class_idx]
    except Exception as e:
        print(f"CNN prediction error: {e}")
        return 'standing'

def detect_pose(
    detector: vision.PoseLandmarker,
    frame: np.ndarray,
    timestamp_ms: int,
    visibility_threshold: float = 0.55,
) -> Tuple[Optional[object], Dict[str, Tuple[int, int]]]:
    """Run pose estimation on a video frame."""
    
    # Convert to MediaPipe Image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detect
    detection_result = detector.detect_for_video(mp_image, timestamp_ms)

    keypoints: Dict[str, Tuple[int, int]] = {}
    
    # Check if any pose was detected
    if not detection_result.pose_landmarks:
        return detection_result, keypoints

    # Process the first detected pose
    landmarks = detection_result.pose_landmarks[0]
    h, w = frame.shape[:2]

    for name, idx in JOINT_INDICES.items():
        if idx < len(landmarks):
            landmark = landmarks[idx]
            # Visibility/Presence might not be directly available in the unified list same way as legacy logic needed,
            # but usually 'visibility' or 'presence' fields exist.
            # In the new API, landmarks have .visibility and .presence attributes.
            if landmark.visibility >= visibility_threshold:
                 keypoints[name] = (int(landmark.x * w), int(landmark.y * h))

    return detection_result, keypoints

def draw_skeleton(frame: np.ndarray, keypoints: Dict[str, Tuple[int, int]]) -> np.ndarray:
    """Draw keypoints and skeleton edges on the frame."""
    for start_joint, end_joint in SKELETON_EDGES:
        if start_joint in keypoints and end_joint in keypoints:
            cv2.line(frame, keypoints[start_joint], keypoints[end_joint], (255, 180, 0), 2)

    for _, (x, y) in keypoints.items():
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    return frame

def _compute_motion_overlay(
    prev_gray: Optional[np.ndarray],
    curr_bgr: np.ndarray,
    max_corners: int = 120,
) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
    """Estimate simple motion vectors using sparse optical flow with Harris corners."""
    curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY) if len(curr_bgr.shape) == 3 else curr_bgr
    if prev_gray is None:
        return curr_bgr, curr_gray, 0

    # Use Harris corners instead of goodFeaturesToTrack
    prev_pts = detect_harris_corners(prev_gray, max_corners=max_corners)
    
    if prev_pts is None or len(prev_pts) == 0:
        return curr_bgr, curr_gray, 0

    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    if curr_pts is None or status is None:
        return curr_bgr, curr_gray, 0

    good_new = curr_pts[status.flatten() == 1]
    good_old = prev_pts[status.flatten() == 1]

    motion_vectors = 0
    for new, old in zip(good_new, good_old):
        a, b = new.ravel()
        c, d = old.ravel()
        cv2.arrowedLine(curr_bgr, (int(c), int(d)), (int(a), int(b)), (0, 0, 255), 1, tipLength=0.3)
        motion_vectors += 1

    return curr_bgr, curr_gray, motion_vectors

def _parse_source(source: str) -> Union[int, str]:
    return int(source) if source.isdigit() else source

def _toggle_fullscreen(window_name: str, is_fullscreen: bool) -> bool:
    new_state = not is_fullscreen
    if new_state:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    return new_state

def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time human detection and motion analysis system.")
    parser.add_argument("--source", default="0", help="Webcam index or stream URL.")
    parser.add_argument("--save-output", default=None, help="Optional output video path (e.g., out.mp4).")
    parser.add_argument("--print-joints", action="store_true", help="Print joint coordinates to console.")
    parser.add_argument("--optical-flow", action="store_true", help="Overlay basic optical-flow motion vectors.")
    parser.add_argument("--hog-detection", action="store_true", help="Use HOG for human detection.")
    parser.add_argument("--motion-classification", action="store_true", help="Classify motion patterns (walking/running/standing).")
    args = parser.parse_args()

    # Initialize MediaPipe Tasks Landmarker
    try:
        landmarker = load_landmarker(MODEL_PATH)
    except Exception as e:
        print(f"Error initializing MediaPipe: {e}")
        return

    cap = cv2.VideoCapture(_parse_source(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    window_name = "DroidCam Pose Estimation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    is_fullscreen = False

    writer = None
    if args.save_output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps_out = cap.get(cv2.CAP_PROP_FPS)
        fps_out = fps_out if fps_out and fps_out > 1 else 20.0
        writer = cv2.VideoWriter(args.save_output, cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (width, height))

    prev_gray = None
    prev_time = time.time()
    start_time = time.time()
    prev_keypoints = None
    frame_count = 0
    
    # Initialize motion classifier
    motion_classifier = None
    if args.motion_classification:
        motion_classifier = load_or_create_motion_classifier()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame read failed. Exiting...")
                break

            # Calculate timestamp in ms for MediaPipe
            timestamp_ms = int((time.time() - start_time) * 1000)

            processed = preprocess_frame(frame)
            
            # HOG detection if enabled
            hog_boxes = []
            if args.hog_detection:
                hog_boxes = detect_humans_hog(processed)
                for (x, y, w, h) in hog_boxes:
                    cv2.rectangle(processed, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Detect pose using MediaPipe
            _, keypoints = detect_pose(landmarker, processed, timestamp_ms)
            
            output = draw_skeleton(processed, keypoints)
            
            # Motion classification
            motion_label = None
            if args.motion_classification:
                if motion_classifier and TF_AVAILABLE:
                    motion_label = classify_motion_cnn(motion_classifier, processed, keypoints)
                else:
                    motion_label = classify_motion_rule_based(keypoints, prev_keypoints, frame_count)
                
                if motion_label:
                    cv2.putText(output, f"Motion: {motion_label.upper()}", (10, 150), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if args.print_joints and keypoints:
                print(keypoints)
                if motion_label:
                    print(f"Motion: {motion_label}")

            if args.optical_flow:
                output, prev_gray, vector_count = _compute_motion_overlay(prev_gray, output)
                cv2.putText(output, f"Motion vectors: {vector_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            current_time = time.time()
            fps = 1.0 / max(current_time - prev_time, 1e-6)
            prev_time = current_time

            people_count = len(hog_boxes) if args.hog_detection else (1 if keypoints else 0)
            cv2.putText(output, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(output, f"People detected: {people_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(output, "Press 'f' for fullscreen, 'q' to quit", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            
            prev_keypoints = dict(keypoints) if keypoints else None
            frame_count += 1

            cv2.imshow(window_name, output)
            if writer is not None:
                writer.write(output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("f"):
                is_fullscreen = _toggle_fullscreen(window_name, is_fullscreen)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
