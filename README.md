✅ README.md — Jumbled Frames Reconstruction Challenge
Jumbled Frames Reconstruction — 10s Video @ 30 FPS

This project reconstructs the correct temporal order of a jumbled 10-second video (300 frames @ 30 fps).
The reconstruction uses visual similarity metrics, window-based optimization, and a greedy nearest-neighbor ordering strategy.

✅ Features

Frame extraction from input video

Multiple similarity metrics (SSIM, Histogram)

Window-based similarity computation for speed

Multi-processing support

Three reconstruction modes:

Fast (Histogram only)

Balanced (SSIM + Histogram, window=20)

Accurate (Full SSIM, highest similarity score)

Predictable runtime (based on selected mode)

Final reordered video export

Logging + execution summary

✅ Installation
pip install -r requirements.txt

✅ Directory Structure
project/
│   re_construct_optimized.py
│   requirements.txt
│   README.md
│   ALGORITHM.md
│
├── shuffled_test/
│      jumbled_video.mp4
│
└── output/
       frames/
       similarity_matrix.npy
       reconstruction_order.txt
       reconstructed_video.mp4
       execution_summary.txt

✅ Usage
Basic Usage
python re_construct_optimized.py --input shuffled_test/jumbled_video.mp4 --outdir output_fast --fps 30

✅ Recommended Modes
Fast Mode (Testing / Quick Runs)

Histogram-only (very fast)

Good for checking pipeline

python re_construct_optimized.py --input shuffled_test/jumbled_video.mp4 --outdir output_fast --mode fast

Balanced Mode (Final Submission Recommended)

SSIM + Histogram

Window size = 20

Best trade-off between accuracy and speed

python re_construct_optimized.py --mode balanced --window 20

Accurate Mode (Highest Similarity Score)

Full SSIM

Perfect for final evaluation

python re_construct_optimized.py --mode accurate

✅ Output

reconstructed_video.mp4 → Final reordered output video

execution_summary.txt → Total time, frame count, mode used

reconstruction_order.txt → Ordered list of frame indices

similarity_matrix.npy → Saved similarity matrix (optional)

✅ How It Works (Short Overview)

Extract frames

Downscale for faster processing

Compute similarity between each frame and its neighbors (window-based)

Build a similarity graph

Pick a start-frame with lowest average similarity

Greedy nearest-neighbor ordering

Rebuild the video using new order

Full explanation available in ALGORITHM.md.

✅ Requirements

All dependencies are listed in:

requirements.txt

✅ Notes

Designed for 10s @ 30fps (≈300 frames)

Window size affects both speed & accuracy

Multiprocessing allows large speedups

For best results, use balanced or accurate mode

✅ Author

Submission for TEC-DIA Jumbled Frame Reconstruction (Round 1)
