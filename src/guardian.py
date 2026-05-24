"""
Aerial Guardian - Person Detection and Tracking Pipeline
Author: Abhay Kumar Yadav
Date: May 2026

Pipeline:
1. YOLOv8n for detection (lightweight, 6MB)
2. ByteTrack for multi-object tracking
3. Trajectory tails for visual history
"""

import cv2
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def draw_trajectory(frame, trajectory, color):
    if len(trajectory) < 2:
        return
    for i in range(1, len(trajectory)):
        alpha = i / len(trajectory)
        thickness = max(1, int(3 * alpha))
        faded_color = tuple(int(c * alpha) for c in color)
        cv2.line(frame, trajectory[i-1], trajectory[i], faded_color, thickness)


def run_tracker(video_path, output_path, conf=0.25, max_tail=40):
    print("Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps_video}fps, {total_frames} frames")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps_video, (width, height))
    
    trajectories = {}
    id_colors = {}
    frame_times = []
    frame_count = 0
    
    print("Starting tracking pipeline...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        t0 = time.time()
        frame_count += 1
        
        results = model.track(
            frame,
            classes=[0],
            tracker="bytetrack.yaml",
            persist=True,
            conf=conf,
            imgsz=640,
            verbose=False
        )[0]
        
        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            
            for box, tid, conf_score in zip(boxes, track_ids, confidences):
                x1, y1, x2, y2 = map(int, box)
                tid = int(tid)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                if tid not in id_colors:
                    np.random.seed(tid * 7)
                    id_colors[tid] = tuple(int(x) for x in np.random.randint(100, 255, 3))
                
                color = id_colors[tid]
                
                if tid not in trajectories:
                    trajectories[tid] = []
                trajectories[tid].append((cx, cy))
                trajectories[tid] = trajectories[tid][-max_tail:]
                
                draw_trajectory(frame, trajectories[tid], color)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"ID:{tid} ({conf_score:.2f})"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        t1 = time.time()
        frame_times.append(t1 - t0)
        
        if len(frame_times) > 30:
            recent_fps = 1 / np.mean(frame_times[-30:])
            cv2.putText(frame, f"FPS: {recent_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        out.write(frame)
        
        if frame_count % 50 == 0:
            avg_fps = 1 / np.mean(frame_times)
            print(f"  Frame {frame_count}/{total_frames} | Avg FPS: {avg_fps:.2f} | Active tracks: {len(trajectories)}")
    
    cap.release()
    out.release()
    
    avg_fps = 1 / np.mean(frame_times)
    print("\n" + "="*50)
    print("PIPELINE COMPLETE")
    print("="*50)
    print(f"Output saved: {output_path}")
    print(f"Total frames: {frame_count}")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Total unique IDs tracked: {len(trajectories)}")
    print(f"Hardware: CPU (MacBook)")
    print("="*50)
    
    return avg_fps


if __name__ == "__main__":
    run_tracker(
        video_path="../demo/test_sequence.avi",
        output_path="../demo/output_tracked.mp4",
        conf=0.25,
        max_tail=40
    )
