import cv2
import os
from pathlib import Path

def sequence_to_video(seq_folder, output_path, fps=30):
    """
    Convert a folder of images into a video file.
    VisDrone gives us image sequences, not videos.
    """
    seq_path = Path(seq_folder)
    frames = sorted(seq_path.glob("*.jpg"))
    
    if not frames:
        print(f"No frames found in {seq_folder}")
        return
    
    # Get dimensions from first frame
    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    print(f"Converting {len(frames)} frames...")
    for i, frame_path in enumerate(frames):
        frame = cv2.imread(str(frame_path))
        out.write(frame)
        if i % 100 == 0:
            print(f"  Processed {i}/{len(frames)} frames")
    
    out.release()
    print(f"Video saved: {output_path}")
    print(f"Duration: {len(frames)/fps:.1f} seconds at {fps} FPS")


if __name__ == "__main__":
    seq_folder = "../data/VisDrone2019-MOT-val/sequences/uav0000086_00000_v"
    output_path = "../demo/test_sequence.mp4"
    
    os.makedirs("../demo", exist_ok=True)
    sequence_to_video(seq_folder, output_path, fps=30)
