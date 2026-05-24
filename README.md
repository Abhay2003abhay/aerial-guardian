# Aerial Guardian
Person Detection and Tracking from Drone Footage

## Results
- Model: YOLOv8n (6.2 MB)
- Baseline detections: 20 persons
- SAHI detections: 39 persons (+95%)
- Total IDs tracked: 48
- Avg FPS: 4.83

## How to Run
pip install ultralytics sahi opencv-python matplotlib
cd src
python3 guardian.py

## Project Structure
notebooks/ - exploration notebooks
src/ - main pipeline code
results/ - sample outputs
data/ - VisDrone dataset
