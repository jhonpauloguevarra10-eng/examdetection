import urllib.request  # type: ignore[import-not-found]
from pathlib import Path

dest = Path.home() / '.mediapipe'
dest.mkdir(parents=True, exist_ok=True)

candidates = {
    'face_landmarker.task': [
        'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker.task',
        'https://storage.googleapis.com/mediapipe-models/vision_landmark_detector/face_landmarker/float16/1/face_landmarker.task',
        'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker.task?raw=true'
    ],
    'hand_landmarker.task': [
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker.task',
        'https://storage.googleapis.com/mediapipe-models/vision_landmark_detector/hand_landmarker/float16/1/hand_landmarker.task',
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker.task?raw=true'
    ]
}

results = {}
for name, urls in candidates.items():
    saved = False
    for u in urls:
        out = dest / name
        try:
            print(f"Trying: {u}")
            urllib.request.urlretrieve(u, out)
            print(f"Saved {name} to {out}")
            results[name] = str(out)
            saved = True
            break
        except Exception as e:
            print(f"Failed: {u} -> {e}")
    if not saved:
        results[name] = None

print('\nSummary:')
for k, v in results.items():
    if v:
        print(f"{k}: downloaded -> {v}")
    else:
        print(f"{k}: NOT downloaded. Please download manually and place it in {dest}")
