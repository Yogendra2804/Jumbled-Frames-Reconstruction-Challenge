# 🧩 Jumbled Video Frame Reconstruction — 300 Frames (5s @ 60 FPS)

This project reconstructs the correct temporal order of a **jumbled 300-frame video**  
(≈5 seconds @ 60 fps).  
The pipeline uses **SSIM + HSV Histogram similarity**, **window-based optimization**, and a  
**greedy nearest-neighbor ordering algorithm** to restore the original sequence.

---

## ✅ Drive Link (Required for Evaluation)
https://drive.google.com/drive/folders/16sAugEmChvkVtMbp52JPm1ZRHERbBkdd?usp=drive_link

→ Please check the Drive folder for **input video**, **reconstructed output video**, and **sample runs**.  
(GitHub cannot preview .mp4 videos.)

---

## ✅ Features

- 🔍 **Frame extraction** from input video  
- 📊 **Multiple similarity metrics** (SSIM + HSV Histogram)  
- ⚡ **Window-based similarity computation** (massively faster than full O(N²))  
- 🧵 **Multi-processing** for parallel computation  
- 🎛️ **Adjustable reconstruction modes**  
  - **Fast** → Histogram-only  
  - **Balanced** → SSIM + Histogram (window=20) **(recommended)**  
  - **Accurate** → Large window (highest similarity accuracy)  
- 🎞 **Final reordered video output**  
- 📝 **Logging + execution summary**  

---

## ✅ Installation

Make sure Python **3.8+** is installed.

```bash
pip install -r requirements.txt
```

---

## ✅ Directory Structure

```
project/
│
├── re_construct_optimized.py
├── re_construct.py
├── shuffle_frames.py
│
├── README.md
├── Algorithm.md
├── requirements.txt
│
├── shuffled_test/
│     └── jumbled_video.mp4
│
├── output_fast/
│     ├── reconstructed.mp4
│     ├── summary.txt
│     └── frames/
│
└── Videos/
      ├── Input_Sample.mp4
      └── Reconstructed_Sample.mp4
```

---

## ✅ Usage

### ▶️ Basic Command
```bash
python re_construct_optimized.py --input shuffled_test/jumbled_video.mp4 --outdir output_fast --fps 60
```

---

## ✅ Modes

### ⚡ Fast Mode (Testing / Debugging)
- Small window  
- Very fast  
- Ideal for pipeline checks  

```
python re_construct_optimized.py --window 8 --workers 6
```

---

### ✅ Balanced Mode (Recommended for Submission)
- Window = 20  
- SSIM + Histogram  
- Best **accuracy vs speed** ratio  

```
python re_construct_optimized.py --window 20 --workers 10
```

---

### 🎯 Accurate Mode (Maximum Precision)
- Window = 30  
- More comparisons  
- Best reconstruction accuracy  

```
python re_construct_optimized.py --window 30 --workers 12
```

---

## ✅ Output Files
reconstructed_video.mp4 ->	Final reordered video
reconstruction_order.txt -> 	Ordered list of frame indices
execution_summary.txt ->	Processing time, settings used

---

## ✅ How It Works — Short Overview

1. Extract frames from jumbled video  
2. Downscale frames for fast comparison  
3. Compute similarity within a **local window**  
4. Build similarity matrix  
5. Choose “start frame” using lowest similarity score  
6. Apply **Greedy Nearest-Neighbor** ordering  
7. Reassemble output video in predicted order  

→ Full technical explanation is available in **Algorithm.md**

---

## ✅ Requirements

All dependencies are included in:

```
requirements.txt
```

---

## ✅ Notes

- Designed for **300-frame** videos (5 seconds @ 60 fps)  
- Window size heavily affects accuracy & runtime  
- Multi-processing significantly speeds up similarity matrix computation  
- Balanced mode is ideal for real evaluation conditions  

---

## ✅ Author

**Yogendra Gupta**  
Submission for **TEC-DIA — Jumbled Frames Reconstruction Challenge (Round 1)**  
VIT Vellore  

