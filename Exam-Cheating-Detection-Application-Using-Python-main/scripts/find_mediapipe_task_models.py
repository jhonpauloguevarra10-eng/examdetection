import mediapipe as mp  # type: ignore[import-not-found]
import os, glob, shutil
from pathlib import Path

pkg_dir = os.path.dirname(mp.__file__)
print('mediapipe package dir:', pkg_dir)
found = glob.glob(os.path.join(pkg_dir, '**', '*.task'), recursive=True)
print('found .task files:', found)

dest = Path.home() / '.mediapipe'
dest.mkdir(parents=True, exist_ok=True)

if not found:
    print('No .task files found inside installed mediapipe package.')

for f in found:
    try:
        dst = dest / os.path.basename(f)
        shutil.copy(f, dst)
        print('copied', f, '->', dst)
    except Exception as e:
        print('failed to copy', f, e)

print('\nDone.')
