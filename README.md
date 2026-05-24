# Aerial Guardian
### Person Detection and Tracking from Drone Footage

Assignment submission for Research Engineer (Computer Vision and Deep Learning)

---

## The Problem
Detecting persons from a moving drone is hard because:
- Targets are tiny (often 10-30 pixels tall)
- Camera is constantly moving, causing ID switching
- Real-time performance needed on lightweight hardware

## My Solution

### Architecture
- Detector: YOLOv8n (nano) — 6.2MB, fast on CPU
- Tracker: ByteTrack — handles occlusion via motion prediction
- Key addition: SAHI tiled inference for small person detection
- Visual feature: Colour-faded trajectory tails per tracked ID

### Why These Choices?
I first ran standard YOLOv8n and found it missed most persons
because they were too small. After exploring the dataset, I saw
most persons were under 30px tall — well below what standard
detectors are trained for.

This led me to integrate SAHI (Slicing Aided Hyper Inference),
which splits each frame into overlapping 320x320 tiles, runs
detection on each tile, then merges results. This significantly
improved detection of small targets.

---

## Results

| Metric | Value |
|--------|-------|
| Model size | 6.2 MB (limit: 300MB) |
| Avg FPS CPU | 4.83 fps |
| Hardware tested | MacBook CPU |
| Tracker | ByteTrack |
| Baseline detections | 20 persons |
| SAHI detections | 39 persons (+95%) |
| Total IDs tracked | 48 unique persons |

### Speed vs Accuracy Trade-off

| Method | FPS | Detection Quality |
|--------|-----|------------------|
| YOLOv8n standard | 4.78 | misses small persons |
| YOLOv8n + SAHI | 0.24 | catches small persons |

---

## How to Run

### Setup
git clone https://github.com/Abhay2003abhay/aerial-guardian.git
cd aerial-guardian
pip install ultralytics sahi opencv-python matplotlib

### Run tracking pipeline
cd src
python3 guardian.py

### Run notebooks in order
1. notebooks/01_data_exploration.ipynb
2. notebooks/02_detection_v1.ipynb
3. notebooks/03_detection_v2_sahi.ipynb
4. notebooks/04_results_summary.ipynb

---

## How I Handled ID Switching
ByteTrack uses a Kalman filter to predict where each person
will be in the next frame even when briefly occluded.
Unlike DeepSORT, it does not need a re-identification network,
making it lighter and faster — better for edge deployment.

## Edge Deployment on Jetson
Export to TensorRT: model.export(format='engine', device=0)
Expected 5-10x FPS improvement on Jetson Nano vs CPU.
SAHI slice size can be increased to 512x512 to reduce
number of tiles and improve FPS with minor accuracy trade-off.

---

## Limitations
- CPU inference is slow (4.83 FPS) — needs GPU for real-time
- SAHI adds latency — tunable via slice size
- Model occasionally loses ID on fast-moving or occluded persons
- Night and low-light performance not tested

## What I Would Do With More Time
- Fine-tune YOLOv8n on VisDrone training set
- Test YOLOv8s for better accuracy within 300MB limit
- Add re-identification features for long-term tracking
- Quantize model to INT8 for faster CPU inference

---

## Project Structure
aerial-guardian/
├── notebooks/               Exploration and experiment notebooks
├── src/
│   ├── guardian.py          Main tracking pipeline
│   └── sequence_to_video.py
├── data/                    VisDrone dataset (not in repo)
├── results/                 Sample outputs and charts
├── demo/                    Output tracked video
└── report/                  PDF report
