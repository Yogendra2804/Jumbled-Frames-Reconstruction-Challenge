#!/usr/bin/env python3
"""
shuffle_frames.py
Create a jumbled test video by shuffling frames of a clean video.

Usage:
python shuffle_frames.py --input clean_video.mp4 --outdir shuffled_out --fps 30

This script:
1. Extracts all frames from the input clean video.
2. Randomly shuffles the frame order.
3. Saves shuffled frames into outdir/frames_shuffled/.
4. Builds a jumbled .mp4 video (jumbled_video.mp4).
"""

import os
import cv2
import random
import argparse
from tqdm import tqdm 

def ensure(path):
    os.makedirs(path, exist_ok=True)

def extract_frames(video_path, out_dir):
    ensure(out_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fname = os.path.join(out_dir, f"frame_{idx:04d}.png")
        cv2.imwrite(fname, frame)
        frames.append(fname)
        idx += 1

    cap.release()
    print(f"[INFO] Extracted {len(frames)} frames.")
    return frames

def write_video(frames_list, video_path, fps):
    if len(frames_list) == 0:
        raise RuntimeError("No frames to write in output video.")

    first_img = cv2.imread(frames_list[0])
    h, w = first_img.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    for fpath in tqdm(frames_list, desc="Writing video"):
        img = cv2.imread(fpath)
        if img is None:
            img = 255 * np.ones((h, w, 3), dtype=np.uint8)
        writer.write(img)

    writer.release()
    print(f"[INFO] Video written: {video_path}")

# Main
def main():
    parser = argparse.ArgumentParser(description="Shuffle frames of a clean video.")
    parser.add_argument("--input", required=True, help="Path to clean input video.")
    parser.add_argument("--outdir", default="shuffled_output", help="Output folder.")
    parser.add_argument("--fps", type=int, default=None, help="Force output FPS (optional).")
    args = parser.parse_args()

    ensure(args.outdir)

    frames_extracted_dir = os.path.join(args.outdir, "frames_extracted")
    frames_shuffled_dir = os.path.join(args.outdir, "frames_shuffled")
    ensure(frames_shuffled_dir)

    # extract frames
    frames = extract_frames(args.input, frames_extracted_dir)

    # get original fps if user didn't force it
    cap = cv2.VideoCapture(args.input)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    fps = args.fps if args.fps else orig_fps
    print(f"[INFO] Output FPS = {fps}")

    # shuffle
    shuffled = frames.copy()
    random.shuffle(shuffled)

    # save shuffled frames
    print("[INFO] Saving shuffled frames...")
    saved_paths = []
    for i, old_path in enumerate(shuffled):
        img = cv2.imread(old_path)
        out_path = os.path.join(frames_shuffled_dir, f"shuffle_{i:04d}.png")
        cv2.imwrite(out_path, img)
        saved_paths.append(out_path)

    print(f"[INFO] Saved {len(saved_paths)} shuffled frames.")

    # write video
    out_video = os.path.join(args.outdir, "jumbled_video.mp4")
    write_video(saved_paths, out_video, fps=fps)

    print("\n✅ Jumbled video created successfully!\n")
    print(f"Jumbled video path: {out_video}")
    print(f"Shuffled frames directory: {frames_shuffled_dir}")

if __name__ == "__main__":
    main()
