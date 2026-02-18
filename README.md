# Air Drawing

Draw in the air using hand gestures. Uses your webcam + mediapipe to track your hands and lets you draw on screen by moving your finger around.

## What it does

- Tracks your hand in real time through the webcam
- Index finger up = draw
- All fingers open = erase
- Index + middle up = change color
- Fist = switch between pen and brush
- Both hands fist = clear canvas
- Supports two hands simultaneously

## Setup

You need Python 3.8+ and a working webcam.

```
pip install opencv-python mediapipe numpy
```

## Run it

```
python main.py
```

Press `q` to quit, `c` to clear the canvas.

## How it works

The project is split into a few parts:

- `core/` - camera handling, hand tracking with mediapipe, gesture detection logic
- `app/` - canvas drawing, tools (pen/brush/eraser), UI overlay, config
- `ml/` - dataset collection and training for custom gesture recognition (optional)

The gesture engine looks at which fingers are up/down and maps that to actions. Theres debounce logic so it doesnt flicker between gestures when youre moving your hand.

## ML stuff (optional)

If you want to train your own gesture classifier instead of using the rule-based one:

1. Collect data: `python -m ml.dataset_collector`
2. Train: `python -m ml.trainer`
3. The model gets saved as `gesture_model.pkl`

Its a simple nearest-centroid classifier - nothing fancy but it gets the job done.

## Project structure

```
core/
  camera.py          - webcam wrapper
  hand_tracker.py    - mediapipe hand detection
  gesture_engine.py  - gesture recognition
  state_manager.py   - tracks current tool/color

app/
  canvas.py          - drawing surface
  tools.py           - pen, brush, eraser
  ui.py              - fps counter, color palette, tool indicator
  config.py          - all the settings

ml/
  dataset_collector.py - collect training data
  trainer.py           - train gesture model
  model_inference.py   - use trained model

main.py              - entry point, ties everything together
```

## Known issues

- Thumb detection can be iffy depending on hand orientation
- Works best with good lighting
- Performance drops if you have a lot of stuff drawn on canvas (lots of blending)

## License

Do whatever you want with it.
