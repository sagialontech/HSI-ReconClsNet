# HSI-ReconClsNet

# Scene-Adaptive Hyperspectral Chemical Identification
### Single-Cube Learning with Physics-Guided Augmentation

This repository contains a **scene-adaptive hyperspectral imaging (HSI) classification pipeline** for detecting chemical contamination from a **single hyperspectral cube**.

The approach is inspired by the human-inspired *machine education* framework proposed by  
*Kendler et al., Scientific Reports (2022)*, but introduces a **conceptual shift**:

> Instead of training a model that generalizes across many scenes,  
> we train a model **adapted to a single scene**.

---

## 📌 Key Idea (TL;DR)

- HSI data is expensive and highly sensitive to illumination and background
- Global generalization across scenes is difficult and data-intensive
- Illumination and background are **fixed within a single cube**
- We therefore **fit the model to one cube** and detect contamination within it

To achieve this, we use a **two-stage neural pipeline**:

1. **ReconNet (U-Net)**  
   Learns scene-specific illumination and background and reconstructs a material-focused spectrum

2. **ClsNet**  
   Classifies pixels using both the original spectrum and the reconstructed output

Each **pixel is treated as a training sample**.

---

## 📂 Repository Structure

The project assumes the following directory structure:

```
project_root/
│
├── HSI-ReconClsNet.ipynb        # Main notebook (entry point)
├── README.md                 # This file
│
├── data/
│   ├── cube_name         # Hyperspectral cube (H × W × Bands)
│   │   ├── dataset_cache # folder contains the model and dataset generated to enable fast reloading
│   │   │   └── ...
│   │   │   # what below here need to be unzipped
│   │   ├── cube.hdr         # cube's header file
│   │   ├── cube.raw         # cube's raw data
│   │   ├── DARKREF_cube.hdr # cube's dark current header file
│   │   ├── DARKREF_cube.raw # cube's dark current raw data
│   │   ├── cube_GT.npz      # cube's ground truth
│   │   └── ...
│   ├── Ref.csv             # reference csv continue the contamination material spectral
│   └── ...
│
├── GT/
│   ├── gt_map_maker.py # script to generate the ground truth (cube_GT.npz) easily 
│   ├── readme_gt_maker.md
│
└── 
```

---

## 🚀 Getting Started

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

---

### 2️⃣ Create environment (recommended)

```bash
conda create -n hsi python=3.9
conda activate hsi
pip install -r requirements.txt
```

---

### 3️⃣ Prepare the data

- Place **one hyperspectral cube** in `data/`
- Cube shape must be: **(Height × Width × SpectralBands)**
- Place **reference spectral signatures** (1D vectors) in `reference_spectra/`

No multi-cube dataset is required.

---

## 🧠 Notebook Usage (`myv3_cleaned.ipynb`)

### Step 1 — Open notebook

```bash
jupyter notebook myv3_cleaned.ipynb
```

---

### Step 2 — Set global configuration

At the top of the notebook, edit the **Globals / Configuration** section:

```python
CUBE_PATH = "data/cube_001.npy"
REFERENCE_SPECTRA_DIR = "reference_spectra/"

TRAIN_RATIO = 0.05
NUM_EPOCHS_RECON = 50
NUM_EPOCHS_CLS = 30

USE_AUGMENTATION = True
DEVICE = "cuda"
```

---

### Step 3 — Run the notebook

Run the notebook **top to bottom**.

Pipeline stages:
1. Load hyperspectral cube
2. Select clean background pixels
3. Generate synthetic contaminated spectra
4. Train ReconNet
5. Reconstruct spectra
6. Train ClsNet
7. Predict contamination map
8. Visualize results

---

## 🔁 Training Paradigm

- **Each pixel = one sample**
- Train/test split is **pixel-wise**
- Only **p% of pixels** are used for training
- Remaining pixels are classified without labels

---

## 🧪 Physics-Guided Augmentation

Training data is generated using:
- Clean background pixels from the same cube
- Reference spectra of target materials
- A nonlinear spectral mixing model

---

## 🧩 Pipeline Summary

```
HSI Cube
   ↓
Physics-Guided Augmentation
   ↓
ReconNet (U-Net)
   ↓
[Original Spectrum + Reconstructed Spectrum]
   ↓
ClsNet
   ↓
Pixel-wise Contamination Map
```

---

## ⚠️ Assumptions & Limitations

**Assumptions**
- Fixed illumination within cube
- Learnable background statistics
- Accurate reference spectra

**Limitations**
- No cross-scene generalization
- Retraining required per cube
- Designed for scene-level analysis

---

## 📈 Recommended Use Cases

✔ Limited HSI scenes  
✔ Strong illumination variability  
✔ Cost-sensitive acquisition  
✔ Rapid deployment needs  

---

## 📝 Citation

If you use this work, please cite:

```
Kendler, S. et al.
Hyperspectral imaging for chemicals identification:
a human-inspired machine learning approach
Scientific Reports, 2022
```

---

## ✨ Final Note

This repository prioritizes **clarity, interpretability, and practical deployment**
over large-scale generalization.
