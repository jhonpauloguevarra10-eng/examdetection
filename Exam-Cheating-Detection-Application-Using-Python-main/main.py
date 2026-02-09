import cv2  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]
import time
import mediapipe as mp  # type: ignore[import-not-found]

# Initialize YOLO (detection model) - optional
yolo_model = None
try:
    from ultralytics import YOLO  # type: ignore[import-not-found]
    yolo_model = YOLO('yolov8n.pt')
except ImportError:
    print("Warning: YOLO not available, running without object detection")
print(mp.__version__)

# Cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# MediaPipe initialization (Face Mesh for accurate iris/eye tracking, Hands for finger detection)
mp_face_mesh = None
mp_hands = None
mp_drawing = None

try:
    # MediaPipe Tasks API (newer version - primary approach)
    from mediapipe.tasks.python import vision  # type: ignore[import-not-found]
    from mediapipe.tasks.python.core import base_options  # type: ignore[import-not-found]
    
    # Get model paths from ~/.mediapipe directory
    import os
    from pathlib import Path
    mediapipe_models_dir = Path.home() / '.mediapipe'
    face_model_path = mediapipe_models_dir / 'face_landmarker.task'
    hand_model_path = mediapipe_models_dir / 'hand_landmarker.task'
    
    # Check if models exist
    if face_model_path.exists() and hand_model_path.exists():
        # Initialize FaceLandmarker for iris/eye detection
        base_opts = base_options.BaseOptions(model_asset_path=str(face_model_path))
        face_options = vision.FaceLandmarkerOptions(base_options=base_opts, num_faces=2)
        mp_face_mesh = vision.FaceLandmarker.create_from_options(face_options)
        
        # Initialize HandLandmarker for finger tracking
        base_opts_hand = base_options.BaseOptions(model_asset_path=str(hand_model_path))
        hand_options = vision.HandLandmarkerOptions(base_options=base_opts_hand, num_hands=2)
        mp_hands = vision.HandLandmarker.create_from_options(hand_options)
        
        mp_drawing = vision.drawing_utils
        print("[INFO] MediaPipe Tasks API loaded successfully (using downloaded models)")
    else:
        print(f"[INFO] MediaPipe models not found in {mediapipe_models_dir}. Set mp as None (will use Haar cascades)")
        mp_face_mesh = None
        mp_hands = None
        mp_drawing = None
except ImportError as e:
    print(f"[INFO] MediaPipe import error: {e}. Falling back to Haar cascades only.")
    mp_face_mesh = None
    mp_hands = None
    mp_drawing = None
except Exception as e:
    print(f"[INFO] MediaPipe Tasks initialization error: {e}. Falling back to Haar cascades only.")
    mp_face_mesh = None
    mp_hands = None
    mp_drawing = None

# Tracking variables
away_count = 0
phone_detected_count = 0
unauthorized_person_detected_count = 0
head_pose_violations = 0
body_movement_violations = 0
iris_movement_violations = 0
eye_looking_away_count = 0
finger_movement_violations = 0
hand_near_face_count = 0
gesture_violations_count = 0  # Track hand/finger gestures
individual_finger_tracking = {  # Track individual finger states
    'THUMB_EXT': 0, 'INDEX_EXT': 0, 'MIDDLE_EXT': 0, 'RING_EXT': 0, 'PINKY_EXT': 0
}
gesture_types = {  # Track specific gestures detected
    'PEACE_SIGN': 0, 'THUMBS_UP': 0, 'POINTING': 0, 'CLOSED_FIST': 0, 'OPEN_HAND': 0, 'THREE_FINGERS': 0
}
total_frames = 0
cheating_detected_frames = 0

# Store previous frame data for movement detection
previous_positions = {}
previous_iris_positions = {}
previous_hand_positions = {}
movement_threshold = 50  # pixels
iris_movement_threshold = 15  # pixels for iris movement

# Head pose calibration
calibrated_head_position = None
calibrated_iris_position = None

def detect_iris_movement(frame, face_x, face_y, face_w, face_h):
    """Detect iris/eye gaze movement and suspicious behavior"""
    violations = []
    iris_positions = []
    
    # Try MediaPipe Face Landmarks first for accurate iris detection
    violations_mp = []
    iris_positions_mp = []
    if mp_face_mesh is not None:
        try:
            import cv2 as cv
            from mediapipe.framework.formats import mp_image  # type: ignore[import-not-found]
            
            # Convert frame to MP Image format (RGB)
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_img = mp_image.MPImage(image_format=mp_image.ImageFormat.SRGB, data=rgb_frame)
            
            # Run face landmarker
            face_detection_result = mp_face_mesh.detect(mp_img)
            
            if face_detection_result.face_landmarks:
                h, w = frame.shape[:2]
                for face_landmarks in face_detection_result.face_landmarks:
                    # Use landmarks to find iris landmarks (indices 468-472 and 473-477)
                    if len(face_landmarks) > 477:
                        # left iris (468-472)
                        l_iris = [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in range(468, 473)]
                        r_iris = [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in range(473, 478)]
                        if l_iris:
                            lx = sum(p[0] for p in l_iris) // len(l_iris)
                            ly = sum(p[1] for p in l_iris) // len(l_iris)
                            iris_positions_mp.append((lx, ly))
                        if r_iris:
                            rx = sum(p[0] for p in r_iris) // len(r_iris)
                            ry = sum(p[1] for p in r_iris) // len(r_iris)
                            iris_positions_mp.append((rx, ry))

                        # Estimate gaze by comparing iris center to eye corner landmarks
                        try:
                            left_outer = face_landmarks[33]
                            left_inner = face_landmarks[133]
                            right_outer = face_landmarks[362]
                            right_inner = face_landmarks[263]
                            # compute approximate eye boxes
                            lox, loy = int(left_outer.x * w), int(left_outer.y * h)
                            lix, liy = int(left_inner.x * w), int(left_inner.y * h)
                            rox, roy = int(right_outer.x * w), int(right_outer.y * h)
                            rix, riy = int(right_inner.x * w), int(right_inner.y * h)

                            # check for extreme horizontal iris offsets
                            for (ix, iy) in iris_positions_mp:
                                # choose nearest eye box
                                if abs(ix - (lox + lix)//2) < abs(ix - (rox + rix)//2):
                                    eye_w = abs(lox - lix) or 1
                                    offset_x = (ix - min(lox, lix)) / eye_w
                                else:
                                    eye_w = abs(rox - rix) or 1
                                    offset_x = (ix - min(rox, rix)) / eye_w

                                if offset_x < 0.15 or offset_x > 0.85:
                                    violations_mp.append({'type': 'IRIS_SIDE', 'x1': face_x, 'y1': face_y-40, 'x2': face_x+face_w, 'y2': face_y+face_h, 'message': 'EYES LOOKING SIDEWAYS - CHEATING', 'confidence': 0.95})
                                # downward gaze
                                if iy > face_y + face_h * 0.65:
                                    violations_mp.append({'type': 'IRIS_DOWN', 'x1': face_x, 'y1': face_y-40, 'x2': face_x+face_w, 'y2': face_y+face_h, 'message': 'EYES LOOKING DOWN - SUSPICIOUS', 'confidence': 0.9})
                        except Exception:
                            pass
        except Exception:
            violations_mp = []
            iris_positions_mp = []

    if violations_mp:
        return violations_mp, iris_positions_mp

    # Fallback to Haar cascades if MediaPipe didn't detect
    face_roi = frame[face_y:face_y+face_h, face_x:face_x+face_w]
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    # Detect eyes in face
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(eyes) >= 2:
        # Get the two largest eyes
        eyes = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[:2]
        
        for eye in eyes:
            eye_x, eye_y, eye_w, eye_h = eye
            
            # Extract eye region
            eye_roi = gray[eye_y:eye_y+eye_h, eye_x:eye_x+eye_w]
            
            # Apply thresholding to find iris (dark region)
            _, thresh = cv2.threshold(eye_roi, 60, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour (iris)
                iris_contour = max(contours, key=cv2.contourArea)
                iris_center = cv2.moments(iris_contour)
                
                if iris_center['m00'] > 0:
                    iris_cx = int(iris_center['m10'] / iris_center['m00'])
                    iris_cy = int(iris_center['m01'] / iris_center['m00'])
                    
                    # Convert to frame coordinates
                    iris_frame_x = face_x + eye_x + iris_cx
                    iris_frame_y = face_y + eye_y + iris_cy
                    iris_positions.append((iris_frame_x, iris_frame_y))
                    
                    # Check iris position within eye (gaze direction)
                    iris_offset_ratio_x = iris_cx / eye_w
                    iris_offset_ratio_y = iris_cy / eye_h
                    
                    # If iris is at extreme edges, student is looking away
                    if iris_offset_ratio_x < 0.2 or iris_offset_ratio_x > 0.8:
                        violations.append({
                            'type': 'IRIS_SIDE',
                            'x1': face_x, 'y1': face_y - 40, 'x2': face_x + face_w, 'y2': face_y + face_h,
                            'message': 'EYES LOOKING SIDEWAYS - CHEATING',
                            'confidence': 0.9
                        })
                    
                    if iris_offset_ratio_y > 0.7:
                        violations.append({
                            'type': 'IRIS_DOWN',
                            'x1': face_x, 'y1': face_y - 40, 'x2': face_x + face_w, 'y2': face_y + face_h,
                            'message': 'EYES LOOKING DOWN - SUSPICIOUS',
                            'confidence': 0.85
                        })
    
    elif len(eyes) == 1:
        # One eye only detected - might be suspicious
        violations.append({
            'type': 'ONE_EYE',
            'x1': face_x, 'y1': face_y - 40, 'x2': face_x + face_w, 'y2': face_y + face_h,
            'message': 'PARTIAL FACE DETECTED - LOOKING AWAY',
            'confidence': 0.7
        })
    
    else:
        # No eyes detected - face turned away
        violations.append({
            'type': 'NO_EYES',
            'x1': face_x, 'y1': face_y - 40, 'x2': face_x + face_w, 'y2': face_y + face_h,
            'message': 'FACE TURNED AWAY - CHEATING',
            'confidence': 0.95
        })
    
    return violations, iris_positions

def detect_head_pose(frame, results):
    """Detect head position and estimate pose"""
    height, width = frame.shape[:2]
    violations = []
    
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls = result.names[int(box.cls[0])]
                if cls == "person":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    face_width = x2 - x1
                    face_height = y2 - y1
                    face_center_x = (x1 + x2) // 2
                    face_center_y = (y1 + y2) // 2
                    
                    # Check if head is turned away (left/right)
                    frame_center_x = width // 2
                    horizontal_offset = abs(face_center_x - frame_center_x)
                    
                    if horizontal_offset > width * 0.25:  # Head turned more than 25% away
                        violations.append({
                            'type': 'HEAD_TURNED',
                            'x1': x1, 'y1': y1-40, 'x2': x2, 'y2': y2,
                            'message': 'LOOKING AWAY',
                            'confidence': horizontal_offset / (width * 0.5)
                        })
                    
                    # Check if head is tilted down (looking at phone/paper)
                    frame_center_y = height // 2
                    vertical_offset = face_center_y - frame_center_y
                    
                    if vertical_offset > height * 0.2:  # Head tilted down
                        violations.append({
                            'type': 'HEAD_DOWN',
                            'x1': x1, 'y1': y1-40, 'x2': x2, 'y2': y2,
                            'message': 'LOOKING DOWN - SUSPICIOUS',
                            'confidence': vertical_offset / (height * 0.5)
                        })
    
    return violations


def analyze_individual_fingers(hand_landmarks):
    """
    Analyze each of the 5 fingers individually for pose and extension
    Returns dict with finger states: {thumb, index, middle, ring, pinky}
    Each finger has: tips_pos, is_extended, is_bent, is_curled, flexion_angle
    """
    finger_bones = {
        'thumb': [0, 1, 2, 3, 4],      # wrist to thumb tip
        'index': [0, 5, 6, 7, 8],      # wrist to index tip
        'middle': [0, 9, 10, 11, 12],  # wrist to middle tip
        'ring': [0, 13, 14, 15, 16],   # wrist to ring tip
        'pinky': [0, 17, 18, 19, 20]   # wrist to pinky tip
    }
    
    fingertip_indices = [4, 8, 12, 16, 20]
    pip_indices = [3, 6, 10, 14, 18]  # Proximal interphalangeal joint
    mcp_indices = [2, 5, 9, 13, 17]   # Metacarpophalangeal joint
    
    # Palm center for distance calculation
    palm_center_x = (hand_landmarks[5].x + hand_landmarks[9].x + hand_landmarks[13].x + hand_landmarks[17].x) / 4
    palm_center_y = (hand_landmarks[5].y + hand_landmarks[9].y + hand_landmarks[13].y + hand_landmarks[17].y) / 4
    
    finger_states = {}
    
    for finger_name, bone_indices in finger_bones.items():
        tip_idx = bone_indices[-1]
        pip_idx = bone_indices[-2]
        mcp_idx = bone_indices[-3]
        
        tip = hand_landmarks[tip_idx]
        pip = hand_landmarks[pip_idx]
        mcp = hand_landmarks[mcp_idx]
        
        # Distance from fingertip to palm
        dist_to_palm = ((tip.x - palm_center_x)**2 + (tip.y - palm_center_y)**2) ** 0.5
        
        # Flexion angle: angle between MCP-PIP-TIP (joints)
        # Calculate vectors
        v1_x = mcp.x - pip.x
        v1_y = mcp.y - pip.y
        v2_x = tip.x - pip.x
        v2_y = tip.y - pip.y
        
        # Dot product and magnitude
        dot_product = v1_x * v2_x + v1_y * v2_y
        mag1 = (v1_x**2 + v1_y**2) ** 0.5
        mag2 = (v2_x**2 + v2_y**2) ** 0.5
        
        if mag1 > 0 and mag2 > 0:
            flexion_angle = dot_product / (mag1 * mag2)  # Cosine of angle
        else:
            flexion_angle = 0
        
        # State determination
        is_extended = dist_to_palm > 0.08
        is_curled = dist_to_palm < 0.04
        is_bent = not is_extended and not is_curled
        
        finger_states[finger_name] = {
            'tip': (int(tip.x * 1000), int(tip.y * 1000)),  # Store normalized coords
            'pip': (int(pip.x * 1000), int(pip.y * 1000)),
            'mcp': (int(mcp.x * 1000), int(mcp.y * 1000)),
            'dist_to_palm': dist_to_palm,
            'is_extended': is_extended,
            'is_bent': is_bent,
            'is_curled': is_curled,
            'flexion_angle': flexion_angle,
            'all_bones': bone_indices
        }
    
    return finger_states


def draw_finger_skeleton(frame, hand_landmarks, finger_states, w, h):
    """
    Draw enhanced hand skeleton with all 5 fingers color-coded
    Shows bone joints and finger states
    """
    # Colors for each finger
    finger_colors = {
        'thumb': (255, 0, 0),      # Blue
        'index': (0, 255, 0),      # Green
        'middle': (0, 0, 255),     # Red
        'ring': (255, 255, 0),     # Cyan
        'pinky': (255, 0, 255)     # Magenta
    }
    
    # Draw each finger with state indicators
    for finger_name, state in finger_states.items():
        bone_indices = state['all_bones']
        color = finger_colors[finger_name]
        
        # Draw bones (connections between joints)
        for i in range(len(bone_indices) - 1):
            start_idx = bone_indices[i]
            end_idx = bone_indices[i + 1]
            
            start_pt = (int(hand_landmarks[start_idx].x * w), int(hand_landmarks[start_idx].y * h))
            end_pt = (int(hand_landmarks[end_idx].x * w), int(hand_landmarks[end_idx].y * h))
            
            # Line thickness based on flexion
            thickness = 3 if state['is_extended'] else (2 if state['is_bent'] else 1)
            cv2.line(frame, start_pt, end_pt, color, thickness)
        
        # Draw joints as circles
        for idx in bone_indices:
            pt = (int(hand_landmarks[idx].x * w), int(hand_landmarks[idx].y * h))
            
            if idx == bone_indices[0]:  # Wrist/base
                radius = 4
                cv2.circle(frame, pt, radius, color, -1)
            elif idx == bone_indices[-1]:  # Fingertip
                radius = 5 if state['is_extended'] else (3 if state['is_bent'] else 2)
                cv2.circle(frame, pt, radius, color, -1)
            else:  # Middle joints
                radius = 3
                cv2.circle(frame, pt, radius, (200, 200, 200), -1)
        
        # Draw state indicator above fingertip
        tip_x = int(hand_landmarks[bone_indices[-1]].x * w)
        tip_y = int(hand_landmarks[bone_indices[-1]].y * h)
        
        state_text = "EXT" if state['is_extended'] else ("BND" if state['is_bent'] else "CRL")
        state_color = (0, 255, 0) if state['is_extended'] else ((255, 165, 0) if state['is_bent'] else (0, 0, 255))
        cv2.putText(frame, state_text, (tip_x - 15, tip_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, state_color, 1)
    
    return frame


def detect_five_finger_combinations(finger_states):
    """
    Detect gestures and poses from all 5 finger states
    Returns list of detected combinations
    """
    violations = []
    
    extended_fingers = {name: state['is_extended'] for name, state in finger_states.items()}
    bent_fingers = {name: state['is_bent'] for name, state in finger_states.items()}
    curled_fingers = {name: state['is_curled'] for name, state in finger_states.items()}
    
    num_extended = sum(extended_fingers.values())
    num_bent = sum(bent_fingers.values())
    num_curled = sum(curled_fingers.values())
    
    # Individual finger recognition
    finger_status = []
    for name, is_ext in extended_fingers.items():
        if is_ext:
            finger_status.append(f"{name.upper()}_EXT")
    
    # Gesture combinations
    if extended_fingers['thumb'] and extended_fingers['index'] and not extended_fingers['middle'] and not extended_fingers['ring'] and not extended_fingers['pinky']:
        violations.append({'gesture': 'PEACE_SIGN', 'confidence': 0.9})
    
    if extended_fingers['thumb'] and not extended_fingers['index'] and not extended_fingers['middle'] and not extended_fingers['ring'] and not extended_fingers['pinky']:
        violations.append({'gesture': 'THUMBS_UP', 'confidence': 0.85})
    
    if not extended_fingers['thumb'] and extended_fingers['index'] and not extended_fingers['middle'] and not extended_fingers['ring'] and not extended_fingers['pinky']:
        violations.append({'gesture': 'POINTING', 'confidence': 0.8})
    
    if num_extended == 0 and num_curled > 2:
        violations.append({'gesture': 'CLOSED_FIST', 'confidence': 0.85})
    
    if num_extended >= 4:
        violations.append({'gesture': 'OPEN_HAND', 'confidence': 0.75})
    
    if extended_fingers['middle'] and extended_fingers['ring'] and extended_fingers['pinky'] and not extended_fingers['index'] and not extended_fingers['thumb']:
        violations.append({'gesture': 'THREE_FINGERS', 'confidence': 0.8})
    
    return violations, finger_status


def detect_hand_gestures(hand_landmarks, wrist_x, wrist_y, face_x, face_y, face_w, face_h):
    """Detect suspicious hand gestures (peace sign, thumbs up, fist, phone holding, etc.)"""
    violations = []
    
    if not hand_landmarks or len(hand_landmarks) < 21:
        return violations
    
    # Extract key landmarks
    # Finger tips: thumb(4), index(8), middle(12), ring(16), pinky(20)
    # PIP joints: thumb(3), index(6), middle(10), ring(14), pinky(18)
    # MCP joints: thumb(2), index(5), middle(9), ring(13), pinky(17)
    # Wrist: 0, Palm base: 5,9,13,17
    
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    middle_tip = hand_landmarks[12]
    ring_tip = hand_landmarks[16]
    pinky_tip = hand_landmarks[20]
    
    thumb_pip = hand_landmarks[3]
    index_pip = hand_landmarks[6]
    middle_pip = hand_landmarks[10]
    ring_pip = hand_landmarks[14]
    pinky_pip = hand_landmarks[18]
    
    palm_center_x = (hand_landmarks[5].x + hand_landmarks[9].x + hand_landmarks[13].x + hand_landmarks[17].x) / 4
    palm_center_y = (hand_landmarks[5].y + hand_landmarks[9].y + hand_landmarks[13].y + hand_landmarks[17].y) / 4
    
    # Calculate distances from fingertips to palm center (extended = far from palm)
    def distance_to_palm(tip):
        return ((tip.x - palm_center_x)**2 + (tip.y - palm_center_y)**2) ** 0.5
    
    thumb_dist = distance_to_palm(thumb_tip)
    index_dist = distance_to_palm(index_tip)
    middle_dist = distance_to_palm(middle_tip)
    ring_dist = distance_to_palm(ring_tip)
    pinky_dist = distance_to_palm(pinky_tip)
    
    # Determine which fingers are extended (threshold for extension)
    extension_threshold = 0.08  # Normalized distance threshold
    thumb_extended = thumb_dist > extension_threshold
    index_extended = index_dist > extension_threshold
    middle_extended = middle_dist > extension_threshold
    ring_extended = ring_dist > extension_threshold
    pinky_extended = pinky_dist > extension_threshold
    
    extended_fingers = [thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]
    num_extended = sum(extended_fingers)
    
    # Check for closed fist (no fingers extended, all fingers curled toward palm)
    all_fingers_close = all(d < 0.05 for d in [thumb_dist, index_dist, middle_dist, ring_dist, pinky_dist])
    if all_fingers_close and num_extended == 0:
        violations.append({
            'type': 'CLOSED_FIST',
            'x1': wrist_x - 40, 'y1': wrist_y - 40,
            'x2': wrist_x + 40, 'y2': wrist_y + 40,
            'message': 'CLOSED FIST - SUSPICIOUS',
            'confidence': 0.85
        })
    
    # Check for peace/victory sign (only index and middle extended)
    if extended_fingers == [False, True, True, False, False]:
        violations.append({
            'type': 'PEACE_SIGN',
            'x1': wrist_x - 50, 'y1': wrist_y - 50,
            'x2': wrist_x + 50, 'y2': wrist_y + 50,
            'message': 'PEACE SIGN - SUSPICIOUS GESTURE',
            'confidence': 0.9
        })
    
    # Check for thumbs up (only thumb extended, pointing up)
    if extended_fingers == [True, False, False, False, False]:
        # Check if thumb is pointing up (thumb_tip.y < middle_pip.y)
        if thumb_tip.y < middle_pip.y - 0.1:
            violations.append({
                'type': 'THUMBS_UP',
                'x1': wrist_x - 40, 'y1': wrist_y - 60,
                'x2': wrist_x + 40, 'y2': wrist_y + 20,
                'message': 'THUMBS UP - SUSPICIOUS',
                'confidence': 0.85
            })
    
    # Check for pointing gesture (only index extended)
    if extended_fingers == [False, True, False, False, False]:
        violations.append({
            'type': 'POINTING',
            'x1': wrist_x - 40, 'y1': wrist_y - 40,
            'x2': wrist_x + 40, 'y2': wrist_y + 40,
            'message': 'POINTING GESTURE - SUSPICIOUS',
            'confidence': 0.8
        })
    
    # Check for open hand (all fingers extended - common for distraction/talking)
    if num_extended >= 4:
        violations.append({
            'type': 'OPEN_HAND',
            'x1': wrist_x - 50, 'y1': wrist_y - 50,
            'x2': wrist_x + 50, 'y2': wrist_y + 50,
            'message': f'OPEN HAND GESTURE ({num_extended} fingers)',
            'confidence': 0.75
        })
    
    # Check for phone holding pose (thumb and index pinched, hand tilted, held near face)
    thumb_index_close = abs(thumb_tip.x - index_tip.x) < 0.05 and abs(thumb_tip.y - index_tip.y) < 0.05
    phone_distance = ((wrist_x - face_x)**2 + (wrist_y - face_y)**2) ** 0.5
    
    if thumb_index_close and phone_distance < 200:
        violations.append({
            'type': 'PHONE_HOLDING',
            'x1': max(0, wrist_x - 60), 'y1': max(0, wrist_y - 80),
            'x2': wrist_x + 60, 'y2': wrist_y + 80,
            'message': 'PHONE HOLDING POSE - CHEATING',
            'confidence': 0.95
        })
    
    # Check for OK sign (thumb and index forming circle, other fingers extended)
    if extended_fingers == [False, False, True, True, True]:
        violations.append({
            'type': 'OK_SIGN',
            'x1': wrist_x - 45, 'y1': wrist_y - 45,
            'x2': wrist_x + 45, 'y2': wrist_y + 45,
            'message': 'OK SIGN - SUSPICIOUS SIGNAL',
            'confidence': 0.8
        })
    
    return violations


def detect_hands_movement(frame, face_centers=None):
    """Detect hand/finger presence and suspicious movements using MediaPipe Hands with finger bone tracking"""
    violations = []
    hand_positions = []
    finger_movements = []
    
    if mp_hands is None:
        return violations, hand_positions
    
    try:
        from mediapipe.framework.formats import mp_image  # type: ignore[import-not-found]
        
        # Convert frame to MP Image format (RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp_image.MPImage(image_format=mp_image.ImageFormat.SRGB, data=rgb_frame)
        
        # Run hand landmarker
        hand_detection_result = mp_hands.detect(mp_img)
        
        if hand_detection_result.hand_landmarks:
            h, w = frame.shape[:2]
            for hand_idx, hand_landmarks in enumerate(hand_detection_result.hand_landmarks):
                # Extract all 21 hand landmarks (wrist + finger joints)
                hand_points = []
                for landmark_idx, landmark in enumerate(hand_landmarks):
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    hand_points.append((x, y, landmark_idx))
                
                hand_positions.append(hand_points)
                
                # Wrist position (landmark 0)
                wrist_x, wrist_y = hand_points[0][0], hand_points[0][1]
                
                # Analyze all 5 fingers individually
                finger_states = analyze_individual_fingers(hand_landmarks)
                
                # Draw enhanced finger skeleton with state indicators
                frame = draw_finger_skeleton(frame, hand_landmarks, finger_states, w, h)
                
                # Detect gesture combinations from finger analysis
                gesture_combos, finger_status_list = detect_five_finger_combinations(finger_states)
                
                # Create violation entries for detected combinations
                for gesture_info in gesture_combos:
                    gesture = gesture_info['gesture']
                    confidence = gesture_info['confidence']
                    
                    # Map to violation message with finger status
                    gesture_messages = {
                        'PEACE_SIGN': 'PEACE SIGN - SUSPICIOUS',
                        'THUMBS_UP': 'THUMBS UP - CHEATING SIGNAL',
                        'POINTING': 'POINTING GESTURE',
                        'CLOSED_FIST': 'CLOSED FIST - SUSPICIOUS',
                        'OPEN_HAND': 'OPEN HAND - DISTRACTED',
                        'THREE_FINGERS': 'THREE FINGERS EXTENDED'
                    }
                    
                    violations.append({
                        'type': gesture,
                        'x1': wrist_x - 60, 'y1': wrist_y - 60,
                        'x2': wrist_x + 60, 'y2': wrist_y + 60,
                        'message': gesture_messages.get(gesture, 'SUSPICIOUS GESTURE'),
                        'confidence': confidence
                    })
                
                # Detect hand gestures including phone holding pose (legacy check)
                if face_centers and len(face_centers) > 0:
                    face_center = face_centers[0]
                    face_x, face_y = face_center
                    # Check for phone holding pose specifically
                    thumb_tip = hand_landmarks[4]
                    index_tip = hand_landmarks[8]
                    thumb_index_close = abs(thumb_tip.x - index_tip.x) < 0.05 and abs(thumb_tip.y - index_tip.y) < 0.05
                    phone_distance = ((wrist_x - face_x)**2 + (wrist_y - face_y)**2) ** 0.5
                    
                    if thumb_index_close and phone_distance < 200 and not any(g['gesture'] == 'PEACE_SIGN' for g in gesture_combos):
                        violations.append({
                            'type': 'PHONE_HOLDING',
                            'x1': max(0, wrist_x - 60), 'y1': max(0, wrist_y - 80),
                            'x2': wrist_x + 60, 'y2': wrist_y + 80,
                            'message': 'PHONE HOLDING POSE - CHEATING',
                            'confidence': 0.95
                        })
                
                # Calculate finger movement and suspicious activity
                hand_id = f"hand_{hand_idx}"
                
                # Track finger motion (distance between consecutive frames)
                if hand_id in previous_hand_positions:
                    prev_points = previous_hand_positions[hand_id]
                    total_movement = 0
                    
                    for i, (curr_x, curr_y, _) in enumerate(hand_points):
                        if i < len(prev_points):
                            prev_x, prev_y = prev_points[i][:2]
                            dist = ((curr_x - prev_x)**2 + (curr_y - prev_y)**2) ** 0.5
                            total_movement += dist
                    
                    avg_movement = total_movement / len(hand_points)
                    finger_movements.append(avg_movement)
                    
                    # Threshold: lots of finger/hand movement is suspicious (typing gesture)
                    if avg_movement > 20:  # High finger activity
                        violations.append({
                            'type': 'FINGER_MOVEMENT',
                            'x1': wrist_x - 50, 'y1': wrist_y - 50,
                            'x2': wrist_x + 50, 'y2': wrist_y + 50,
                            'message': f'FINGER MOVEMENT - TYPING ({int(avg_movement)}px)',
                            'confidence': min(avg_movement / 50, 1.0)
                        })
                
                previous_hand_positions[hand_id] = hand_points
                
                # Detect hand near face (cheating indicator)
                if face_centers:
                    for (fx, fy) in face_centers:
                        dist = ((wrist_x - fx)**2 + (wrist_y - fy)**2) ** 0.5
                        if dist < 150:
                            violations.append({
                                'type': 'HAND_NEAR_FACE',
                                'x1': max(0, fx - 80), 'y1': max(0, fy - 120),
                                'x2': fx + 80, 'y2': fy + 120,
                                'message': 'HAND NEAR FACE - CHEATING',
                                'confidence': 0.95
                            })
                else:
                    # Hand raised above mid-screen (flagged as suspicious)
                    if wrist_y < frame.shape[0] * 0.45:
                        violations.append({
                            'type': 'HAND_RAISED',
                            'x1': wrist_x - 30, 'y1': wrist_y - 30,
                            'x2': wrist_x + 30, 'y2': wrist_y + 30,
                            'message': 'HAND RAISED - SUSPICIOUS',
                            'confidence': 0.9
                        })
                
                # Detect rapid finger clicking/typing (checking distance between fingertips and palm)
                # Index fingertip (8), middle fingertip (12), ring fingertip (16), pinky fingertip (20)
                fingertips = [hand_points[8], hand_points[12], hand_points[16], hand_points[20]]
                palm_center = ((hand_points[5][0] + hand_points[9][0] + hand_points[13][0] + hand_points[17][0]) // 4,
                               (hand_points[5][1] + hand_points[9][1] + hand_points[13][1] + hand_points[17][1]) // 4)
                
                fingers_extended = 0
                for tip_x, tip_y, _ in fingertips:
                    dist_to_palm = ((tip_x - palm_center[0])**2 + (tip_y - palm_center[1])**2) ** 0.5
                    if dist_to_palm > 50:  # Finger extended
                        fingers_extended += 1
                
                # Multiple fingers extended rapidly could indicate suspicious activity
                if fingers_extended >= 3:
                    violations.append({
                        'type': 'FINGERS_EXTENDED',
                        'x1': wrist_x - 60, 'y1': wrist_y - 60,
                        'x2': wrist_x + 60, 'y2': wrist_y + 60,
                        'message': f'MULTIPLE FINGERS EXTENDED - SUSPICIOUS ({fingers_extended})',
                        'confidence': 0.85
                    })
    
    except Exception as e:
        pass
    
    return violations, hand_positions

def detect_body_movement(frame, results, prev_positions):
    """Detect abnormal body movements"""
    violations = []
    current_positions = {}
    
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            for idx, box in enumerate(result.boxes):
                cls = result.names[int(box.cls[0])]
                if cls == "person":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    current_positions[idx] = (center_x, center_y)
                    
                    # Compare with previous position
                    if idx in prev_positions:
                        prev_x, prev_y = prev_positions[idx]
                        movement = np.sqrt((center_x - prev_x)**2 + (center_y - prev_y)**2)
                        
                        if movement > movement_threshold:
                            violations.append({
                                'type': 'MOVEMENT',
                                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                'message': f'MOVEMENT DETECTED ({int(movement)}px)',
                                'confidence': min(movement / (movement_threshold * 3), 1.0)
                            })
    
    return violations, current_positions

def draw_violation_box(frame, violation):
    """Draw red box for policy violations"""
    x1, y1, x2, y2 = violation['x1'], violation['y1'], violation['x2'], violation['y2']
    message = violation['message']
    
    # Draw thick red border
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    
    # Draw diagonal lines
    cv2.line(frame, (x1, y1), (x1+30, y1+30), (0, 0, 255), 2)
    cv2.line(frame, (x2, y1), (x2-30, y1+30), (0, 0, 255), 2)
    cv2.line(frame, (x1, y2), (x1+30, y2-30), (0, 0, 255), 2)
    cv2.line(frame, (x2, y2), (x2-30, y2-30), (0, 0, 255), 2)
    
    # Add label box on top
    label_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.rectangle(frame, (x1, y1-35), (x1 + label_size[0] + 10, y1-5), (0, 0, 255), -1)
    cv2.putText(frame, message, (x1 + 5, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

def draw_cheating_border(frame, is_cheating):
    """Draw animated border lines when cheating is detected"""
    height, width = frame.shape[:2]
    thickness = 3
    color = (0, 0, 255)  # Red for cheating
    
    if is_cheating:
        cv2.line(frame, (0, 0), (width, 0), color, thickness)  # Top
        cv2.line(frame, (0, height-1), (width, height-1), color, thickness)  # Bottom
        cv2.line(frame, (0, 0), (0, height), color, thickness)  # Left
        cv2.line(frame, (width-1, 0), (width-1, height), color, thickness)  # Right
        
        cv2.line(frame, (0, 0), (50, 50), color, 2)
        cv2.line(frame, (width-50, 0), (width, 50), color, 2)
        cv2.line(frame, (0, height-50), (50, height), color, 2)
        cv2.line(frame, (width-50, height), (width, height-50), color, 2)

# Start capturing
cap = cv2.VideoCapture(0)

# === PRE-CALIBRATION TIMER AND MESSAGE ===
for i in range(5, 0, -1):
    ret, frame = cap.read()
    if not ret:
        break
    cv2.putText(frame, "EXAM PROCTORING SYSTEM", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, "Starting in: " + str(i), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, "WARNING: Cheating will be recorded!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Proctoring System", frame)
    cv2.waitKey(1000)

# === MAIN MONITORING LOOP ===
start_session_time = time.time()
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    total_frames = frame_count
    height, width, _ = frame.shape
    
    phone_detected = False
    cheating_violations = []
    
    # Iris detection using MediaPipe Face Mesh (run every frame for accurate eye tracking)
    face_centers = []
    faces = []
    if mp_face_mesh is not None:
        try:
            from mediapipe.framework.formats import mp_image  # type: ignore[import-not-found]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp_image.MPImage(image_format=mp_image.ImageFormat.SRGB, data=rgb)
            results_mp = mp_face_mesh.detect(mp_img)
            if results_mp and results_mp.face_landmarks:
                h, w = frame.shape[:2]
                for face_landmarks in results_mp.face_landmarks:
                    # approximate bounding box from landmarks
                    xs = [lm.x for lm in face_landmarks]
                    ys = [lm.y for lm in face_landmarks]
                    min_x, max_x = int(min(xs) * w), int(max(xs) * w)
                    min_y, max_y = int(min(ys) * h), int(max(ys) * h)
                    fw, fh = max_x - min_x, max_y - min_y
                    faces.append((min_x, min_y, fw, fh))
                    face_centers.append((min_x + fw//2, min_y + fh//2))
        except Exception:
            faces = []
            face_centers = []

    # If no faces from MediaPipe fall back to Haar cascade
    if not faces:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (face_x, face_y, face_w, face_h) in faces:
        # Detect iris/eye movement for each detected face (MediaPipe inside function)
        iris_violations, iris_positions = detect_iris_movement(frame, face_x, face_y, face_w, face_h)
        cheating_violations.extend(iris_violations)
        if iris_violations:
            iris_movement_violations += len(iris_violations)

    # Hand/finger detection and movement
    if mp_hands is not None:
        hand_violations, hand_positions = detect_hands_movement(frame, face_centers)
        cheating_violations.extend(hand_violations)
        if hand_violations:
            # Count different types of hand violations
            for v in hand_violations:
                if v['type'] == 'FINGER_MOVEMENT':
                    finger_movement_violations += 1
                elif v['type'] == 'HAND_NEAR_FACE':
                    hand_near_face_count += 1
                elif v['type'] in ['CLOSED_FIST', 'PEACE_SIGN', 'THUMBS_UP', 'POINTING', 'OPEN_HAND', 'PHONE_HOLDING', 'OK_SIGN', 'THREE_FINGERS']:
                    gesture_violations_count += 1
                    # Track specific gesture types
                    if v['type'] in gesture_types:
                        gesture_types[v['type']] += 1
                else:
                    body_movement_violations += 1

    # Run YOLO detection periodically to reduce load
    if frame_count % 30 == 0 and yolo_model is not None:
        try:
            # use direct call which is lighter than the predictor CLI wrapper
            results = yolo_model(frame, conf=0.5, verbose=False)

            # normalize results to single Results object
            if isinstance(results, list) and len(results) > 0:
                result = results[0]
            else:
                result = results
            if result.boxes is not None:
                for box in result.boxes:
                        try:
                            cls_idx = int(box.cls[0])
                            cls = result.names[cls_idx]
                            x1, y1, x2, y2 = map(int, box.xyxy[0])

                            if cls == "cell phone":
                                phone_detected = True
                                cheating_violations.append({
                                    'type': 'PHONE',
                                    'x1': x1, 'y1': y1-40, 'x2': x2, 'y2': y2,
                                    'message': 'PHONE DETECTED - CHEATING',
                                    'confidence': 1.0
                                })
                            elif cls == "book":
                                cheating_violations.append({
                                    'type': 'BOOK',
                                    'x1': x1, 'y1': y1-40, 'x2': x2, 'y2': y2,
                                    'message': 'BOOK DETECTED - UNAUTHORIZED',
                                    'confidence': 0.8
                                })
                            elif cls == "person":
                                # Head pose detection based on person location
                                head_violations = detect_head_pose(frame, [result])
                                cheating_violations.extend(head_violations)
                                if head_violations:
                                    head_pose_violations += len(head_violations)
                                
                                # Body movement detection
                                body_violations, current_pos = detect_body_movement(frame, [result], previous_positions)
                                cheating_violations.extend(body_violations)
                                previous_positions = current_pos
                                if body_violations:
                                    body_movement_violations += len(body_violations)
                        except:
                            pass
        except Exception as e:
            pass
    
    # Update cheating count
    if phone_detected:
        phone_detected_count += 1
    
    if cheating_violations:
        cheating_detected_frames += 1
    
    # Draw all violations
    for violation in cheating_violations:
        draw_violation_box(frame, violation)
    
    # Draw face and eye detection boxes (for visual feedback)
    for (face_x, face_y, face_w, face_h) in faces:
        # Draw green box around detected face
        cv2.rectangle(frame, (face_x, face_y), (face_x+face_w, face_y+face_h), (0, 255, 0), 2)
        
        # Draw green circle to show active monitoring
        cv2.circle(frame, (face_x + face_w//2, face_y + face_h//2), 5, (0, 255, 0), -1)
        cv2.putText(frame, "EYE TRACKING", (face_x, face_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw cheating border and warning banner if violations detected
    if cheating_violations:
        draw_cheating_border(frame, True)
        banner_height = 60
        cv2.rectangle(frame, (0, 0), (width, banner_height), (0, 0, 255), -1)
        cv2.putText(frame, "!!! CHEATING DETECTED !!!", (max(10, width//2 - 200), 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3)
    
    # Display policy at top
    policy_text = "POLICY: No phones, books, suspicious movement, or looking away allowed"
    cv2.putText(frame, policy_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    
    # Add statistics with eye tracking info
    iris_violations_this_frame = sum(1 for v in cheating_violations if v['type'] in ['IRIS_SIDE', 'IRIS_DOWN', 'ONE_EYE', 'NO_EYES'])
    eye_status = "EYES: OK" if iris_violations_this_frame == 0 else "EYES: SUSPICIOUS"
    stats_text = f"Frame: {total_frames} | Violations: {len(cheating_violations)} | {eye_status}"
    cv2.putText(frame, stats_text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Proctoring System", frame)
    
    # Check for exit keys
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()

# === SESSION SUMMARY ===
duration = int(time.time() - start_session_time)
print(f"\n{'='*60}")
print(f"EXAM PROCTORING SYSTEM - SESSION SUMMARY")
print(f"{'='*60}")
print(f"Total Duration: {duration} seconds")
print(f"Total Frames Processed: {total_frames}")
print(f"Frames with Violations: {cheating_detected_frames}")
print(f"\nVIOLATIONS DETECTED:")
print(f"  - Phone Detection: {phone_detected_count}")
print(f"  - Head Pose Violations: {head_pose_violations}")
print(f"  - Body Movement Violations: {body_movement_violations}")
print(f"  - Iris/Eye Movement Violations: {iris_movement_violations}")
print(f"\nFINGER & HAND TRACKING:")
print(f"  - Finger Movement Detected: {finger_movement_violations}")
print(f"  - Hand Near Face: {hand_near_face_count}")
print(f"  - Suspicious Hand Gestures: {gesture_violations_count}")
print(f"\n  GESTURE BREAKDOWN (5-FINGER SKELETON RECOGNITION):")
for gesture, count in gesture_types.items():
    if count > 0:
        print(f"    • {gesture}: {count}")
print(f"\n  Individual Finger States Analyzed:")
for finger, count in individual_finger_tracking.items():
    print(f"    • {finger}: {count} detections")
print(f"\nEYE TRACKING:")
print(f"  - Eyes Looking Away: {eye_looking_away_count}")
print(f"\nVIOLATION RATE: {cheating_detected_frames / max(total_frames, 1) * 100:.2f}%")
print(f"{'='*60}")
