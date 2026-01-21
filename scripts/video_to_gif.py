import cv2
import imageio
import os
import argparse

def convert_to_gif(video_path, gif_path, resize_width=480, fps=10):
    print(f"Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        # Skip frames to reduce size if needed (e.g. keep original fps but sample)
        # Here we rely on writer fps
        
        # Resize
        h, w = frame.shape[:2]
        ratio = resize_width / w
        new_h = int(h * ratio)
        frame = cv2.resize(frame, (resize_width, new_h))
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        
    cap.release()
    
    # Limit to 100 frames max for lightweight GIF
    if len(frames) > 100:
        frames = frames[:100]
        
    print(f"Read {len(frames)} frames. Writing GIF...")
    
    # Write GIF
    imageio.mimsave(gif_path, frames, fps=fps, loop=0)
    print(f"GIF saved to {gif_path}")
    print(f"Size: {os.path.getsize(gif_path) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    convert_to_gif(args.input, args.output)
