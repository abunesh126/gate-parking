# SmartPark

SmartPark is a beginner-friendly Flask, OpenCV, CNN and SQLite parking gate
application. The operator explicitly chooses **Entry** or **Exit**; the system
never infers direction from vehicle movement.

## Reference implementation and dataset

The cloned reference repository was inspected without modification. Its first
notebook uses grayscale/bilateral filtering, Canny edges and quadrilateral
contours for plate localization, then contour-based character segmentation and
a Keras CNN. Its `data.zip` contains 36 labelled classes (`0-9`, `A-Z`), with
864 training and 216 validation images. The provided checkpoint is a weights
artifact from that notebook, not a SmartPark model. SmartPark now reuses that
verified checkpoint for inference; no retraining is performed by the live demo.

## Run

```bat
setup.bat
start.bat
```

Open `http://127.0.0.1:5000`. If no model has been trained, the dashboard
clearly reports that CNN status and recognition returns `model_unavailable`
rather than guessing. A camera is optional for database/API tests.

`setup.bat` creates the application environment in `.venv`, installs the
requirements, and initializes `smartpark/backend/data/parking.db`. The
verified checkpoint is already included under `smartpark/ml/models`; no
training command is required for the demonstration. Run `stop.bat` to stop
only the process started by `start.bat`.

The verified reference checkpoint is copied into `ml/models/my_checkpoint`
for inference. Open `/camera` to discover camera indices, start a selected
camera, capture a frame, and explicitly recognize the captured frame. Recognition
uses the contour detector and the restored CNN; it is not run on every live
frame. The camera API releases devices on stop and suppresses the same
recognized plate for a short configurable cooldown. Camera discovery uses
`cv2-enumerate-cameras` on Windows to retrieve device names and stable paths,
probes DirectShow and Media Foundation, verifies a real frame before reporting
a device, and falls back to OpenCV index scanning when named enumeration is
unavailable. Use Refresh cameras after connecting or starting Iriun.

## ML flow

`preprocessing.py` resizes, grayscales, blurs, thresholds and detects edges.
`plate_detection.py` finds plausible wide plate contours. A contour is a
connected boundary in an edge image. `character_segmentation.py` filters
character bounding boxes and orders them left-to-right. `model_loader.py`
restores the exact reference CNN checkpoint using TensorFlow checkpoint
tensors. `predict.py` returns characters, confidence, visual previews, a
bounding box and an explicit status (`recognized`, `uncertain`, or an
unavailable/detection status). `train_cnn.py` remains available for the
separate training workflow but is not used by the live demo.
Enable "Show detection debug images" on `/camera` to inspect the resized
original, grayscale, blur, Canny edges, morphology, contour candidates,
selected crop, segmented characters and prediction confidence.

For an independent camera diagnostic, run:

```bat
.venv\Scripts\python.exe test_cameras.py
```

The selector uses the returned device identifier rather than assuming that a
device's index is globally stable across Windows backends.

## API and tests

The API includes status, dashboard, camera discovery/connect/disconnect, MJPEG
feed, manual parking mode, non-blocking transaction queue, active vehicles,
history and analytics endpoints. SQLite uses parameterized
queries, transactions and a partial unique index preventing duplicate active
sessions. Transactions are processed sequentially with a five-second interval.

The operator workflow is explicit: select `Entry` or `Exit` on the camera
page. After connection, recognition runs automatically at the configured
interval on the selected camera; the current mode is applied only after a
valid, confident recognition. Capture/recognize remains available for
diagnostic use. Exits calculate and store a parking fee. Settings are updated
through `PUT /api/settings`, including capacity, recognition threshold,
detection interval and duplicate cooldown.

Run the automated checks with:

```bat
python -m pytest -q
```
