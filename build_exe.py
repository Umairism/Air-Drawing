"""
Build script for Air Drawing standalone executable.

Usage:
    python build_exe.py

Prerequisites:
    pip install pyinstaller pillow
    hand_landmarker.task must exist in project root (download from MediaPipe)
"""
import subprocess
import sys
import os
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))

def ensure_model():
    """Download hand_landmarker.task if not present."""
    path = os.path.join(ROOT, "hand_landmarker.task")
    if os.path.exists(path):
        print(f"[OK] hand_landmarker.task found ({os.path.getsize(path)} bytes)")
        return
    url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    )
    print("Downloading hand_landmarker.task ...")
    urllib.request.urlretrieve(url, path)
    print(f"[OK] Downloaded ({os.path.getsize(path)} bytes)")

def ensure_icon():
    """Generate icon if not present."""
    path = os.path.join(ROOT, "air_drawing.ico")
    if os.path.exists(path):
        print(f"[OK] air_drawing.ico found")
        return
    print("Generating icon ...")
    subprocess.check_call([sys.executable, os.path.join(ROOT, "build_icon.py")])
    print("[OK] Icon generated")

def build():
    """Run PyInstaller with the spec file."""
    spec = os.path.join(ROOT, "AirDrawing.spec")
    print("\n=== Building AirDrawing.exe ===\n")
    subprocess.check_call([
        sys.executable, "-m", "PyInstaller",
        "--clean",
        "--noconfirm",
        spec,
    ])
    exe_path = os.path.join(ROOT, "dist", "AirDrawing.exe")
    if os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        print(f"\n[SUCCESS] Built: {exe_path}")
        print(f"          Size:  {size_mb:.1f} MB")
    else:
        print("\n[FAILED] AirDrawing.exe was not created.")
        sys.exit(1)

if __name__ == "__main__":
    os.chdir(ROOT)
    ensure_model()
    ensure_icon()
    build()
