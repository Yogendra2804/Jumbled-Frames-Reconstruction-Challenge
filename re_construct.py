#!/usr/bin/env python3
"""
reconstruct.py
Reconstruct a jumbled 10-second, 30fps (300 frames) video by reordering frames.

Usage examples:
python reconstruct.py --input jumbled_video.mp4 --outdir output --fps 30
python reconstruct.py --frames_dir frames_jumbled --outdir output --fps 30

The script:
1. Extracts frames (if given a video).
2. Computes lightweight features & similarity between frames:
   - SSIM on resized grayscale
   - Color histogram correlation (HSV)
   - ORB descriptor match ratio (fast)
   Combined into a composite similarity score.
3. Orders frames using a greedy nearest-neighbor approach (start chosen heuristically).
4. Writes reconstructed .mp4 at specified fps and logs execution time.
"""

import os
import cv2
import numpy as np
import argparse
import time
import logging
from skimage.metrics import structural_similarity as ssim
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

# -------------------------
# Utility functions
# -------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def extract_frames_from_video(video_path, frames_out_dir, expected_total=300):
    ensure_dir(frames_out_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")
    idx = 0
    saved_files = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fname = os.path.join(frames_out_dir, f"frame_{idx:04d}.png")
        cv2.imwrite(fname, frame)
        saved_files.append(fname)
        idx += 1
    cap.release()
    if expected_total and idx != expected_total:
        logging.warning(f"Extracted {idx} frames, but expected {expected_total}. Continue anyway.")
    return saved_files

def load_frames_from_dir(frames_dir, max_expected=300):
    files = [os.path.join(frames_dir, f) for f in sorted(os.listdir(frames_dir)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    files = sorted(files)
    if len(files) == 0:
        raise RuntimeError("No image frames found in directory.")
    # Limit to max_expected if too many (but we expect 300)
    if len(files) > max_expected:
        logging.warning(f"Found {len(files)} frames, trimming to {max_expected}")
        files = files[:max_expected]
    return files

# -------------------------
# Feature / similarity
# -------------------------
def compute_orb_descriptor(img_gray):
    orb = cv2.ORB_create(500)
    kps, des = orb.detectAndCompute(img_gray, None)
    return des

def hist_hsv_similarity(img_bgr, bins=(32,)):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Hue histogram (flatten)
    hist = cv2.calcHist([hsv], [0], None, [bins[0]], [0, 180])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def compare_pair(idx_pair, frames_cache, small_size=(256, 256)):
    i, j = idx_pair
    img_i = frames_cache[i]
    img_j = frames_cache[j]

    # Resize for fast processing
    ii = cv2.resize(img_i, small_size, interpolation=cv2.INTER_AREA)
    jj = cv2.resize(img_j, small_size, interpolation=cv2.INTER_AREA)

    # SSIM on grayscale (range -1..1)
    gi = cv2.cvtColor(ii, cv2.COLOR_BGR2GRAY)
    gj = cv2.cvtColor(jj, cv2.COLOR_BGR2GRAY)
    try:
        ssim_val = ssim(gi, gj, data_range=gi.max() - gi.min())
    except Exception:
        ssim_val = 0.0

    # Color histogram similarity (correlation between hist vectors)
    hi = hist_hsv_similarity(ii, bins=(32,))
    hj = hist_hsv_similarity(jj, bins=(32,))
    # Use correlation (dot / norms)
    denom = (np.linalg.norm(hi) * np.linalg.norm(hj))
    hist_corr = float(np.dot(hi, hj) / denom) if denom > 0 else 0.0

    # ORB matching ratio
    des_i = compute_orb_descriptor(gi)
    des_j = compute_orb_descriptor(gj)
    match_ratio = 0.0
    if des_i is not None and des_j is not None and len(des_i) > 0 and len(des_j) > 0:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        try:
            matches = bf.match(des_i, des_j)
            if matches:
                # use quality: number matches normalized by descriptor count
                match_ratio = float(len(matches) * 2) / (len(des_i) + len(des_j))
                match_ratio = min(match_ratio, 1.0)
        except Exception:
            match_ratio = 0.0

    # Combine scores (weights tuned heuristically)
    # ssim (strong), hist_corr (medium), match_ratio (medium)
    combined = 0.55 * ssim_val + 0.30 * hist_corr + 0.15 * match_ratio
    # Ensure in [0,1]
    combined = float(max(0.0, min(1.0, combined)))
    return (i, j, combined)

# -------------------------
# Similarity matrix calculation
# -------------------------
def compute_similarity_matrix(frames_list, n_workers=None):
    n = len(frames_list)
    logging.info(f"Computing similarity matrix for {n} frames (this may take time).")
    # Load small-res frames into memory for speed
    frames_cache = []
    for fpath in frames_list:
        img = cv2.imread(fpath)
        if img is None:
            raise RuntimeError(f"Failed to read {fpath}")
        frames_cache.append(img)

    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((i, j))

    n_workers = n_workers or max(1, cpu_count() - 1)
    func = partial(compare_pair, frames_cache=frames_cache)
    sim_mat = np.zeros((n, n), dtype=np.float32)

    with Pool(processes=n_workers) as pool:
        for (i, j, score) in tqdm(pool.imap_unordered(func, pairs), total=len(pairs), desc="pairwise"):
            sim_mat[i, j] = score
            sim_mat[j, i] = score

    # Diagonal = 1.0
    np.fill_diagonal(sim_mat, 1.0)
    return sim_mat

# -------------------------
# Ordering algorithm
# -------------------------
def greedy_nearest_neighbor_order(sim_mat):
    """
    Simple greedy ordering: pick a start frame, repeatedly append the most similar unused neighbor.
    For single-shot smooth videos this usually works well.
    """
    n = sim_mat.shape[0]
    unused = set(range(n))

    # Heuristic for start: frame with smallest sum similarity -> likely beginning (less similar to others)
    sums = sim_mat.sum(axis=1)
    start = int(np.argmin(sums))
    order = [start]
    unused.remove(start)

    current = start
    while unused:
        # find the most similar frame among unused
        candidates = list(unused)
        sims = sim_mat[current, candidates]
        # choose the argmax
        next_idx = candidates[int(np.argmax(sims))]
        order.append(next_idx)
        unused.remove(next_idx)
        current = next_idx

    return order

# -------------------------
# Video write
# -------------------------
def write_video_from_order(frames_list, order, out_video_path, fps=30):
    # Use frame size from first frame
    first_img = cv2.imread(frames_list[order[0]])
    h, w = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))
    for idx in order:
        img = cv2.imread(frames_list[idx])
        if img is None:
            logging.warning(f"Missing frame for index {idx}, writing a black frame.")
            img = np.zeros((h, w, 3), dtype=np.uint8)
        writer.write(img)
    writer.release()

# -------------------------
# Main
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Reconstruct jumbled frames for 10s@30fps video.")
    parser.add_argument('--input', required=True, help="Input video file OR frames directory.")
    parser.add_argument('--frames_dir', default=None, help="(Optional) Directory with frames already extracted.")
    parser.add_argument('--outdir', default='output', help="Output directory")
    parser.add_argument('--fps', type=int, default=30, help="Output FPS (default 30)")
    parser.add_argument('--expected_frames', type=int, default=300, help="Expected total frames (default 300)")
    parser.add_argument('--workers', type=int, default=None, help="Number of parallel workers to compute similarity.")
    parser.add_argument('--skip_similarity', action='store_true', help="If set, skip heavy similarity and output frames in original order.")
    return parser.parse_args()

def main():
    args = parse_args()
    ensure_dir(args.outdir)
    logfile = os.path.join(args.outdir, 'reconstruct.log')
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(logfile), logging.StreamHandler()])

    t0 = time.time()
    logging.info("Starting reconstruction pipeline.")
    frames_dir = args.frames_dir

    # If frames_dir given -> use frames, else if input is directory -> use it, else extract frames from video
    frames_list = []
    if args.frames_dir:
        frames_list = load_frames_from_dir(args.frames_dir, max_expected=args.expected_frames)
    elif os.path.isdir(args.input):
        frames_list = load_frames_from_dir(args.input, max_expected=args.expected_frames)
    else:
        # assume input is video file -> extract frames to outdir/frames
        frames_dir = os.path.join(args.outdir, 'frames_extracted')
        ensure_dir(frames_dir)
        frames_list = extract_frames_from_video(args.input, frames_dir, expected_total=args.expected_frames)

    n_frames = len(frames_list)
    logging.info(f"Frames loaded: {n_frames}")

    if n_frames != args.expected_frames:
        logging.warning(f"Number of frames ({n_frames}) differs from expected ({args.expected_frames}). Proceeding anyway.")

    # Compute similarity matrix
    if args.skip_similarity:
        logging.info("skip_similarity set: using original frame order.")
        order = list(range(n_frames))
    else:
        sim_mat = compute_similarity_matrix(frames_list, n_workers=args.workers)
        logging.info("Similarity matrix computed.")
        order = greedy_nearest_neighbor_order(sim_mat)
        logging.info("Greedy ordering computed.")

    out_video_path = os.path.join(args.outdir, 'reconstructed.mp4')
    logging.info(f"Writing reconstructed video to: {out_video_path}")
    write_video_from_order(frames_list, order, out_video_path, fps=args.fps)

    t1 = time.time()
    elapsed = t1 - t0
    logging.info(f"Reconstruction finished. Total time: {elapsed:.2f} seconds.")
    # Save execution summary
    summary_path = os.path.join(args.outdir, 'execution_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"input: {args.input}\n")
        f.write(f"frames_count: {n_frames}\n")
        f.write(f"fps: {args.fps}\n")
        f.write(f"elapsed_seconds: {elapsed:.2f}\n")
        f.write(f"output_video: {out_video_path}\n")
    logging.info(f"Summary written to {summary_path}")

if __name__ == '__main__':
    main()
