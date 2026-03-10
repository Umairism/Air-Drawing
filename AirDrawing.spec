# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Air Drawing.
Produces a single-file executable with all dependencies bundled.

Usage:
    pyinstaller AirDrawing.spec
"""
import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# ---------- paths ----------
ROOT = os.path.abspath(os.path.dirname(SPEC))

# ---------- mediapipe data & submodules ----------
mediapipe_datas = collect_data_files("mediapipe")
mediapipe_hiddenimports = collect_submodules("mediapipe")

# ---------- model file ----------
hand_model = os.path.join(ROOT, "hand_landmarker.task")
if not os.path.exists(hand_model):
    raise FileNotFoundError(
        "hand_landmarker.task not found in project root. Download it from:\n"
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    )

# ---------- data files to bundle ----------
datas = [
    (hand_model, "."),  # model file at root of bundle
] + mediapipe_datas

# optional: bundle a trained ML model if it exists
gesture_model = os.path.join(ROOT, "gesture_model.pkl")
if os.path.exists(gesture_model):
    datas.append((gesture_model, "."))

# ---------- hidden imports ----------
hidden_imports = [
    "cv2",
    "numpy",
    "mediapipe",
    "sklearn",
    "sklearn.svm",
    "sklearn.neighbors",
    "sklearn.preprocessing",
    "sklearn.model_selection",
    "sklearn.utils._typedefs",
    "sklearn.utils._heap",
    "sklearn.utils._sorting",
    "sklearn.utils._vector_sentinel",
    "sklearn.neighbors._partition_nodes",
    "app",
    "app.canvas",
    "app.config",
    "app.tools",
    "app.ui",
    "core",
    "core.camera",
    "core.hand_tracker",
    "core.gesture_engine",
    "core.state_manager",
    "core.noise_filter",
    "core.profiler",
    "ml",
    "ml.model_inference",
    "ml.gesture_customizer",
    "ml.benchmark",
    "ml.dataset_collector",
    "ml.trainer",
] + mediapipe_hiddenimports

# ---------- analysis ----------
a = Analysis(
    [os.path.join(ROOT, "main.py")],
    pathex=[ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "pytest",
        "IPython",
        "jupyter",
        "notebook",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="AirDrawing",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # keep console so user sees status messages
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(ROOT, "air_drawing.ico"),
    version=os.path.join(ROOT, "version_info.txt"),
)
