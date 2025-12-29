# GT Map Maker – Interactive Hyperspectral Annotation Tool

This script (`gt_map_maker.py`) is an interactive tool for creating **pixel-level ground truth (GT) maps**
for hyperspectral ENVI images using manual polygon annotation.

It is designed for building high-quality labeled datasets for material / contamination detection pipelines.

---

## What this tool does

- Loads a hyperspectral ENVI image (`.hdr + .img/.raw`)
- Displays a false-color RGB visualization
- Allows polygon-based annotation of:
  - **Surfaces**: cardboard, wood, ceramics
  - **Contaminants**: PS, silicon, sugar
- Enforces:
  - Contaminants must lie **inside** their surface
  - No overlap between different contaminants
- Exports pixel-level GT maps + visual preview

---

## Installation

Requires Python **3.8+**.

Install dependencies:

```bash
pip install numpy matplotlib spectral
```

On Linux you may also need:

```bash
sudo apt install python3-tk
```

---

## Input data format

You must have an ENVI hyperspectral image:

```
your_image.hdr
your_image.img   (or .raw / .dat)
```

---

## How to run

From the terminal:

```bash
python gt_map_maker.py
```

A file dialog will open — select the `.hdr` file.

---

## Annotation workflow

1. **Select a surface** (cardboard / wood / ceramics)
2. **Draw polygons**
   - Click to add vertices
   - Double-click to close a polygon
   - `z` → undo last polygon
   - `q` → finish current surface
3. Optionally **add contaminant polygons**
   - PS / silicon / sugar
   - Automatically constrained inside the surface
4. Repeat for all surfaces
5. Select **DONE** to generate outputs

You can revisit any surface multiple times.

---

## Output files

Saved next to the input image:

### 1. Ground-truth data
```
your_image_GT.npz
```
Contains:
- `label_map` (H × W pixel labels)
- `wavelengths`
- `id_to_name` mapping

### 2. Pixel-wise CSV
```
your_image_GT.csv
```
Each row corresponds to one pixel and its label.

### 3. Visual preview
```
your_image_GT_preview.png
```
RGB image with transparent GT overlay and legend.

---

## Label scheme

| ID | Label |
|----|------|
| 1–4 | cardboard_(none / PS / silicon / sugar) |
| 5–8 | wood_(none / PS / silicon / sugar) |
| 9–12 | ceramics_(none / PS / silicon / sugar) |
| 0 | unlabeled |

---

## Loading GT in Python

```python
import numpy as np

gt = np.load("your_image_GT.npz", allow_pickle=True)
label_map = gt["label_map"]
wavelengths = gt["wavelengths"]
mapping = gt["mapping"].item()
```

---

## Notes

- Background pixels remain unlabeled (`0`)
- Overlapping contaminant regions are automatically clipped
- Intended for manual, high-quality GT creation

---

## Usage

Research / dataset annotation.
Modify freely to add materials, substrates, or export formats.
