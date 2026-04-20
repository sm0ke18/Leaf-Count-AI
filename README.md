

# 🌿 Leaf-Count AI: Instance Segmentation & Color Diagnostics

An advanced Computer Vision pipeline that solves the "Canopy Problem" in agriculture. This tool utilizes **YOLOv8-Segmentation** to isolate overlapping leaves and **HSV Spectral Analysis** to quantify plant health based on leaf pigment.
🚀 The Challenge
Counting leaves in a dense canopy is difficult because standard bounding boxes overlap, leading to inaccurate counts. Furthermore, "Healthy" vs "Stressed" classification is often subjective. **LeafGuard AI** solves this by:
1.  Using **Pixel-Perfect Masks** to separate individual leaves.
2.  Using **Mathematical Color Thresholds** to report objective health data.

---

 🛠️ Tech Stack
* **Deep Learning:** YOLOv8m-seg (Ultralytics)
* **Computer Vision:** OpenCV (HSV Color Space)
* **Environment:** Python 3.10+, Kaggle (GPU T4)
* **Data Source:** CVPPP Leaf Segmentation Dataset & Custom Field Images



---
 🧬 How It Works

 Phase 1: Instance Segmentation (The "Eyes")
We deployed a **YOLOv8-Medium Segmentation** model. Unlike standard object detection, this model predicts a binary mask for every object. This allows the system to:
* Identify exactly which pixels belong to which leaf.
* Count overlapping leaves with high precision using an optimized **IOU (Intersection over Union)** threshold of `0.7`.

 Phase 2: HSV Color Analysis (The "Brain")
Once a leaf is isolated, the system "cuts out" the leaf and ignores the background (soil, pots, etc.).
* **Color Conversion:** RGB images are converted to **HSV** (Hue, Saturation, Value) to handle varying light conditions.
* **Pixel Counting:** The system counts pixels within specific Hue ranges:
    * **Green:** Hue 35–85 (Healthy Growth)
    * **Yellow/Brown:** Hue 10–34 (Stess/Necrosis)
* **Dominant Logic:** The system labels the leaf based on the majority pixel count or a 1% "Disease Threshold."



---

## 📊 Sample Output
| Metric | Result |
| :--- | :--- |
| **Total Leaf Count** | 4 |
| **Healthy (Green)** | 3 |
| **Stressed (Yellow)** | 1 |
| **Inference Speed** | ~45ms (GPU) |

---

## 💻 Local Installation

1.  **Clone the Repo:**
    ```bash
    git clone https://github.com/yourusername/leaf-guard-ai.git
    cd leaf-guard-ai
    ```

2.  **Install Dependencies:**
    ```bash
    pip install ultralytics opencv-python numpy matplotlib
    ```

3.  **Run Inference:**
    Place your `best.pt` and `input.jpg` in the folder and run:
    ```python
    python leaf_analyzer.py
    ```

---


