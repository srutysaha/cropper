import io
import os
import zipfile
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "blaze_face_short_range.tflite"
)

MIN_CONFIDENCE = 0.35
TILE_WIDTH_RATIO = 0.40
TILE_HEIGHT_RATIO = 0.40
OVERLAP = 0.40
UPSCALE = 4.0

MIN_FACE_WIDTH_RATIO = 0.008
MIN_FACE_HEIGHT_RATIO = 0.008
MIN_FACE_ASPECT = 0.50
MAX_FACE_ASPECT = 1.70

LEFT_REGION = 0.60
VERY_LEFT_REGION = 0.45
RIGHT_REGION = 0.70

GROUP_IOU_THRESHOLD = 0.15
GROUP_DISTANCE_RATIO = 0.50

PADDING_X = 0.15
PADDING_TOP = 0.45
PADDING_BOTTOM = 0.35

# ============================================================
# CACHED FACE DETECTOR CREATION
# ============================================================
@st.cache_resource
def load_detector():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Face detector model not found at expected path:\n{MODEL_PATH}\n"
            "Make sure 'blaze_face_short_range.tflite' is in the same directory as app.py."
        )
    BaseOptions = mp.tasks.BaseOptions
    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
    RunningMode = mp.tasks.vision.RunningMode
    
    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        min_detection_confidence=MIN_CONFIDENCE
    )
    return FaceDetector.create_from_options(options)

# ============================================================
# LOGIC FUNCTIONS (Adapted from your script)
# ============================================================
def create_tiles(image):
    image_height, image_width = image.shape[:2]
    tile_width = min(max(int(image_width * TILE_WIDTH_RATIO), 100), image_width)
    tile_height = min(max(int(image_height * TILE_HEIGHT_RATIO), 100), image_height)
    
    step_x = max(int(tile_width * (1 - OVERLAP)), 1)
    step_y = max(int(tile_height * (1 - OVERLAP)), 1)
    
    y_positions = list(range(0, max(image_height - tile_height, 0) + 1, step_y))
    last_y = max(image_height - tile_height, 0)
    if not y_positions or y_positions[-1] != last_y:
        y_positions.append(last_y)
        
    x_positions = list(range(0, max(image_width - tile_width, 0) + 1, step_x))
    last_x = max(image_width - tile_width, 0)
    if not x_positions or x_positions[-1] != last_x:
        x_positions.append(last_x)
        
    tiles = []
    for y in y_positions:
        for x in x_positions:
            x2 = min(x + tile_width, image_width)
            y2 = min(y + tile_height, image_height)
            tile = image[y:y2, x:x2]
            if tile.size == 0:
                continue
            tiles.append({"image": tile, "x": x, "y": y})
    return tiles

def detect_tile(tile_info, detector):
    tile = tile_info["image"]
    offset_x, offset_y = tile_info["x"], tile_info["y"]
    tile_height, tile_width = tile.shape[:2]
    
    new_width = max(int(tile_width * UPSCALE), 1)
    new_height = max(int(tile_height * UPSCALE), 1)
    resized = cv2.resize(tile, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    result = detector.detect(mp_image)
    faces = []
    if not result.detections:
        return faces
        
    for detection in result.detections:
        if not detection.categories:
            continue
        confidence = float(detection.categories[0].score)
        if confidence < MIN_CONFIDENCE:
            continue
            
        bbox = detection.bounding_box
        x = int(bbox.origin_x / UPSCALE + offset_x)
        y = int(bbox.origin_y / UPSCALE + offset_y)
        width = int(bbox.width / UPSCALE)
        height = int(bbox.height / UPSCALE)
        
        if width <= 0 or height <= 0:
            continue
            
        aspect = width / height
        if aspect < MIN_FACE_ASPECT or aspect > MAX_FACE_ASPECT:
            continue
            
        if (width / tile_width) < MIN_FACE_WIDTH_RATIO or (height / tile_height) < MIN_FACE_HEIGHT_RATIO:
            continue
            
        faces.append({
            "x": max(0, x),
            "y": max(0, y),
            "width": width,
            "height": height,
            "confidence": confidence
        })
    return faces

def detect_faces(image, detector):
    tiles = create_tiles(image)
    all_faces = []
    for tile in tiles:
        all_faces.extend(detect_tile(tile, detector))
    return all_faces

def face_center(face):
    return (face["x"] + face["width"] / 2, face["y"] + face["height"] / 2)

def center_distance(face1, face2):
    x1, y1 = face_center(face1)
    x2, y2 = face_center(face2)
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def calculate_iou(face1, face2):
    f1_x1, f1_y1 = face1["x"], face1["y"]
    f1_x2, f1_y2 = f1_x1 + face1["width"], f1_y1 + face1["height"]
    f2_x1, f2_y1 = face2["x"], face2["y"]
    f2_x2, f2_y2 = f2_x1 + face2["width"], f2_y1 + face2["height"]
    
    x1 = max(f1_x1, f2_x1)
    y1 = max(f1_y1, f2_y1)
    x2 = min(f1_x2, f2_x2)
    y2 = min(f1_y2, f2_y2)
    
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = face1["width"] * face1["height"]
    area2 = face2["width"] * face2["height"]
    union_area = area1 + area2 - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0

def group_faces(faces):
    if not faces:
        return []
    faces = sorted(faces, key=lambda f: f["confidence"], reverse=True)
    groups = []
    for face in faces:
        placed = False
        face_size = max(face["width"], face["height"])
        distance_threshold = face_size * GROUP_DISTANCE_RATIO
        
        for group in groups:
            rep = group["representative"]
            iou = calculate_iou(face, rep)
            dist = center_distance(face, rep)
            rep_size = max(rep["width"], rep["height"])
            threshold = max(distance_threshold, rep_size * GROUP_DISTANCE_RATIO)
            
            if iou > GROUP_IOU_THRESHOLD or dist < threshold:
                group["faces"].append(face)
                if face["confidence"] > rep["confidence"]:
                    group["representative"] = face
                placed = True
                break
        if not placed:
            groups.append({"representative": face, "faces": [face]})
    return groups

def left_position_score(face, image_width):
    center_x, _ = face_center(face)
    pos = center_x / image_width
    if pos <= VERY_LEFT_REGION:
        return 1.0
    if pos <= LEFT_REGION:
        return 1.0 - 0.35 * ((pos - VERY_LEFT_REGION) / (LEFT_REGION - VERY_LEFT_REGION))
    if pos <= RIGHT_REGION:
        return max(0.20, 0.65 - 0.45 * ((pos - LEFT_REGION) / (RIGHT_REGION - LEFT_REGION)))
    return 0.10

def face_size_score(face, image):
    h, w = image.shape[:2]
    return min((face["width"] * face["height"]) / (w * h) * 25, 1.0)

def select_all_faces(faces, image):
    if not faces:
        return []
    img_h, img_w = image.shape[:2]
    groups = group_faces(faces)
    if not groups:
        return []
        
    candidates = []
    for group in groups:
        face = group["representative"]
        support = len(group["faces"])
        conf = face["confidence"]
        pos_score = left_position_score(face, img_w)
        size_score = face_size_score(face, image)
        
        support_score = 1.0 if support >= 5 else (0.95 if support == 4 else (0.85 if support == 3 else (0.70 if support == 2 else 0.15)))
        left_bonus = 0.20 if pos_score >= 0.90 else (0.10 if pos_score >= 0.70 else (-0.25 if pos_score <= 0.20 else 0.0))
        
        score = conf * 0.40 + support_score * 0.25 + pos_score * 0.25 + size_score * 0.10 + left_bonus
        candidates.append({"face": face, "support": support, "confidence": conf, "position_score": pos_score, "score": score})
        
    candidates.sort(key=lambda c: c["score"], reverse=True)
    reliable = []
    for c in candidates:
        if c["confidence"] < 0.45 and c["support"] < 2:
            continue
        if c["position_score"] <= 0.20 and (c["confidence"] < 0.75 or c["support"] < 3):
            continue
        reliable.append(c["face"])
        
    reliable.sort(key=lambda f: (face_center(f)[1], face_center(f)[0]))
    return reliable

def crop_portrait(image, face):
    h, w = image.shape[:2]
    x, y, fw, fh = face["x"], face["y"], face["width"], face["height"]
    px = max(int(fw * PADDING_X), 1)
    pt = max(int(fh * PADDING_TOP), 1)
    pb = max(int(fh * PADDING_BOTTOM), 1)
    
    x1, y1 = max(0, x - px), max(0, y - pt)
    x2, y2 = min(w, x + fw + px), min(h, y + fh + pb)
    return image[y1:y2, x1:x2]

def force_portrait(cropped):
    if cropped is None or cropped.size == 0:
        return None
    h, w = cropped.shape[:2]
    if w > h:
        cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)
    return cropped

# ============================================================
# STREAMLIT WEB APP USER INTERFACE
# ============================================================
st.set_page_config(page_title="Passport Photo Extractor", page_icon="📸", layout="centered")

st.title("📸 Automated Passport Photo Extractor")
st.markdown("Upload group or single photos below. The app will detect faces, crop them cleanly into individual portrait photos, and package everything into a downloadable ZIP file.")

uploaded_files = st.file_uploader(
    "Choose image files",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("Process Images", type="primary"):
        try:
            detector = load_detector()
        except Exception as e:
            st.error(str(e))
            st.stop()
            
        zip_buffer = io.BytesIO()
        total_extracted = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing ({idx+1}/{len(uploaded_files)}): {uploaded_file.name}")
                
                # Read image file from memory buffer
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if image is None:
                    continue
                    
                filename_base = os.path.splitext(uploaded_file.name)[0]
                
                # Detect and crop
                raw_faces = detect_faces(image, detector)
                selected_faces = select_all_faces(raw_faces, image)
                
                for p_idx, face in enumerate(selected_faces, start=1):
                    cropped = crop_portrait(image, face)
                    cropped = force_portrait(cropped)
                    
                    if cropped is not None and cropped.size > 0:
                        success, encoded_img = cv2.imencode(".jpg", cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        if success:
                            out_name = f"{filename_base}_photo_{p_idx}.jpg"
                            zip_file.writestr(out_name, encoded_img.tobytes())
                            total_extracted += 1
                            
                progress_bar.progress((idx + 1) / len(uploaded_files))
                
        status_text.text("Processing complete!")
        progress_bar.empty()
        
        st.success(f"Successfully extracted {total_extracted} photo(s) from {len(uploaded_files)} image(s)!")
        
        if total_extracted > 0:
            zip_buffer.seek(0)
            st.download_button(
                label="📥 Download Cropped Photos (ZIP)",
                data=zip_buffer,
                file_name="extracted_passport_photos.zip",
                mime="application/zip"
            )
        else:
            st.warning("No faces matching criteria were found in the uploaded images.")