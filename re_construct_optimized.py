#!/usr/bin/env python3
"""
Optimized Video Reconstruction (10s @ 30fps)
- Uses SSIM + HSV Histogram only (fast)
- Resizes frames to 96×96 for similarity
- Uses window-based comparisons (±20 frames)
- Multiprocessing enabled
"""

import os
import cv2
import numpy as np
import argparse
import logging
import time
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count
from skimage.metrics import structural_similarity as ssim

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def extract_frames(video_path, out_dir):
    ensure_dir(out_dir)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fpath = os.path.join(out_dir, f"frame_{idx:04d}.png")
        cv2.imwrite(fpath, frame)
        frames.append(fpath)
        idx += 1

    cap.release()
    return frames

def load_frames_from_dir(frames_dir, max_expected=300):
    files = sorted([os.path.join(frames_dir, f)
                    for f in os.listdir(frames_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if len(files) == 0:
        raise RuntimeError("No frames found in directory.")

    if len(files) > max_expected:
        logging.warning(f"Found {len(files)} > expected {max_expected}. Trimming.")
        files = files[:max_expected]

    return files

# ---------------------------------------------------------
# Similarity
# ---------------------------------------------------------
def hist_hsv_similarity(img_bgr, bins=32):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def compare_pair(pair, frames_cache, small_size=(96, 96)):
    i, j = pair
    img_i = cv2.resize(frames_cache[i], small_size)
    img_j = cv2.resize(frames_cache[j], small_size)

    # SSIM (grayscale)
    gi = cv2.cvtColor(img_i, cv2.COLOR_BGR2GRAY)
    gj = cv2.cvtColor(img_j, cv2.COLOR_BGR2GRAY)

    try:
        ssim_val = ssim(gi, gj, data_range=gi.max() - gi.min())
    except:
        ssim_val = 0.0

    # HSV histogram correlation
    hi = hist_hsv_similarity(img_i)
    hj = hist_hsv_similarity(img_j)

    denom = (np.linalg.norm(hi) * np.linalg.norm(hj))
    hist_corr = float(np.dot(hi, hj) / denom) if denom > 0 else 0.0

    # Combine
    score = 0.65 * ssim_val + 0.35 * hist_corr
    return (i, j, score)

# ---------------------------------------------------------
# Reconstruction Logic
# ---------------------------------------------------------
def compute_similarity_matrix(frames_list, window=20, workers=None):
    n = len(frames_list)
    logging.info(f"Computing similarity with window={window} for {n} frames.")

    # Load frames
    frames_cache = [cv2.imread(f) for f in frames_list]

    # Build windowed pairs
    pairs = []
    for i in range(n):
        for off in range(-window, window + 1):
            j = i + off
            if 0 <= j < n and j > i:
                pairs.append((i, j))

    logging.info(f"Total comparisons: {len(pairs)}")

    workers = workers or max(1, cpu_count() - 1)
    func = partial(compare_pair, frames_cache=frames_cache)

    # Init similarity matrix
    sim_mat = np.zeros((n, n), dtype=np.float32)

    with Pool(workers) as pool:
        for i, j, s in tqdm(pool.imap_unordered(func, pairs),
                           total=len(pairs), desc="pairwise"):
            sim_mat[i, j] = s
            sim_mat[j, i] = s

    # Self-similarity = 1
    np.fill_diagonal(sim_mat, 1.0)
    return sim_mat

def greedy_order(sim_mat):
    n = len(sim_mat)
    unused = set(range(n))

    # Start frame = one with lowest total similarity (beginning tends to differ most)
    start = int(np.argmin(sim_mat.sum(axis=1)))
    order = [start]
    unused.remove(start)

    current = start

    while unused:
        candidates = list(unused)
        scores = sim_mat[current, candidates]
        nxt = candidates[int(np.argmax(scores))]

        order.append(nxt)
        unused.remove(nxt)
        current = nxt

    return order

def write_video(order, frames_list, out_path, fps=30):
    first = cv2.imread(frames_list[order[0]])
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    for idx in order:
        img = cv2.imread(frames_list[idx])
        writer.write(img)

    writer.release()

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", default="output_opt")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--expected_frames", type=int, default=300)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    ensure_dir(args.outdir)
    log_file = os.path.join(args.outdir, "log.txt")

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(log_file),
                                  logging.StreamHandler()])

    t0 = time.time()
    logging.info("Starting optimized reconstruction...")

    # Load or extract frames
    if os.path.isdir(args.input):
        frames = load_frames_from_dir(args.input, args.expected_frames)
    else:
        frames_dir = os.path.join(args.outdir, "frames")
        frames = extract_frames(args.input, frames_dir)

    logging.info(f"Frames loaded: {len(frames)}")

    # Compute similarity
    sim_mat = compute_similarity_matrix(frames,
                                        window=args.window,
                                        workers=args.workers)

    logging.info("Similarity matrix computed.")

    # Ordering
    order = greedy_order(sim_mat)
    logging.info("Order computed.")

    # Write output video
    out_vid = os.path.join(args.outdir, "reconstructed.mp4")
    write_video(order, frames, out_vid, fps=args.fps)

    elapsed = time.time() - t0
    logging.info(f"✅ DONE in {elapsed:.2f} seconds.")
    logging.info(f"Output saved: {out_vid}")

    # Save summary
    with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
        f.write(f"elapsed_seconds: {elapsed:.2f}\n")
        f.write(f"frames_used: {len(frames)}\n")
        f.write(f"window: {args.window}\n")
        f.write(f"fps: {args.fps}\n")

if __name__ == "__main__":
    main()
