# 🧩 Jumbled Frames Reconstruction — 10s Video @ 30 FPS

This project reconstructs the correct temporal order of a **jumbled 10-second video**  
(≈300 frames @ 30 fps).  
The pipeline uses **similarity-based frame matching**, **local window optimization**, and a  
**greedy nearest-neighbor ordering algorithm** to restore the original sequence.

---

## ✅ Features

- 🔍 **Frame extraction** from input video  
- 📊 **Multiple similarity metrics** (SSIM + Histogram)  
- ⚡ **Window-based similarity computation** for faster execution  
- 🧵 **Multi-processing** for up to 10× speedup  
- 🎛️ **Three reconstruction modes**:
  - **Fast** → Histogram-only (very fast)
  - **Balanced** → SSIM + Histogram + window=20 (**recommended**)
  - **Accurate** → Full SSIM (highest similarity scores)
- 🗂️ **Predictable runtime** based on mode  
- 🎞️ **Final reordered video export**  
- 📝 **Logging + execution summary** for evaluation  

---

## ✅ Installation

Make sure Python 3.8+ is installed.

```bash
pip install -r requirements.txt

```
✅ Directory Structure
project/
│
├── re_construct_optimized.py
├── README.md
├── ALGORITHM.md
├── requirements.txt
│
├── shuffled_test/
│     └── jumbled_video.mp4
│
└── output/
      ├── frames/
      ├── similarity_matrix.npy
      ├── reconstruction_order.txt
      ├── reconstructed_video.mp4
      └── execution_summary.txt
## ✅ Usage

▶️ Basic Command
```
python re_construct_optimized.py --input shuffled_test/jumbled_video.mp4 --outdir output_fast --fps 30
```
## ✅ Modes

⚡ ✅ Fast Mode (Testing / Debugging)

1. Use small window

2. Fewer comparisons → Much faster

3. ```
   python re_construct_optimized.py --window 8 --workers 6
    ```

✅ Balanced Mode (Recommended for Submission)

1. Uses window=20

2. SSIM + Histogram

3. Best trade-off between speed and accuracy

4.
   ```
   python re_construct_optimized.py --window 20 --workers 10
  ```

✅ Accurate Mode (Slowest but Most Accurate)

1. Uses a large window

2. More comparisons

3. Best reconstruction quality

4. ```
   python re_construct_optimized.py --window 30 --workers 12
   ```

## ✅ Output Files
reconstructed_video.mp4 ->	Final reordered video
reconstruction_order.txt -> 	Ordered list of frame indices
execution_summary.txt ->	Processing time, settings used


## ✅ How It Works — Short Overview
1. Extract all frames from the input jumbled video

2. Downscale frames for faster processing

3. Compute similarity only within a local window (speed optimization)

4. Create a similarity graph

5. Choose starting frame based on lowest global similarity

6. Apply Greedy Nearest-Neighbor ordering

7. Rebuild final video using reordered frame indices

-> Full technical explanation available in ALGORITHM.md.

## ✅ Requirements
All dependencies are included in:
requirements.txt

## ✅ Notes
Designed specifically for 10-second videos @ 30 fps (≈300 frames)

Window size influences speed vs accuracy

Multi-processing drastically reduces runtime

Balanced mode provides the best performance/accuracy ratio

## ✅ Author
Submission for TEC-DIA — Jumbled Frames Reconstruction Challenge (Round 1)
