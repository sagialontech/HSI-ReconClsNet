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
│   │   │   
│   │   ├── cube.hdr         # cube's header file
│   │   ├── cube.raw         # cube's raw data - need to be gunzip from git
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
clone https://github.com/sagialontech/HSI-ReconClsNet.git
cd HSI-ReconClsNet
```

---

### 2️⃣ Create workspace - We used google colab

the notebook is ready to be uploaded as a colab notebook
but can be run from any workspace as long globals and filesystem is correct

---

### 3️⃣ Prepare the data

Look on the example under `data/ ` folder
if you wish to use one of our cube just guzip the .raw splited files

if you want your own cubes you need:
- Place ** hyperspectral cube** in `data/`
- Cube shape must be: **(Height × Width × SpectralBands)**
- Place `Ref.csv`
- the format of the cube need to be ENVI hyperspectral image.


---

## 🧠 Notebook Usage (`HSI-ReconClsNet.ipynb`)

### Step 1 — Open notebook

```bash
jupyter notebook HSI-ReconClsNet.ipynb
```
or
Upload to Colab

---

### Step 2 — Set global configuration

At the top of the notebook, edit the **Globals / Configuration** section:

```python
BASE_DIR   = r"/content/drive/MyDrive/HSI"      # <-- EDIT TO YOUR PATH
DATA_DIR   = rf"{BASE_DIR}/data"
CUBE_NAME = "08-32-58"
...
```

you can change the hyperparamters from the golbal dict HP:
```python
# Model / training hyperparameters
HP ={
    # data related hyperparameters
    "CLEAN_TRAIN_FRACTION": 0.3, # Fraction of pixels assumed to be clean background and used to seed physics-guided augmentation.
    "N_SAMPLES": 10000, # Number of synthetic training samples generated from the mixing model.

    # training related hyperparameters
    "BATCH_SIZE": 64,  # Number of pixel spectra per training batch.
    "STEP_LR_SIZE": 15, # Epoch interval for learning-rate decay.
    "STEP_LR_GAMMA": 0.85, # Learning-rate decay factor.

    # Recon train
    "EPOCHS": 50,
    "LR": 1e-4,
    "LAMBDA_CORR": 1.0,

    # Cls train
    "EPOCHS_CLS": 50,
    "LR_CLS": 1e-4,

    # visual
    "CORR_TH": 0.3, # Correlation threshold for the Naive baseline part
    "K_SMOOTH": 7, # Kernel size for spatial smoothing. 
    "K_BLOCK" : 3, # Block size for local aggregation.
    "TRESH_BLOCK": 0.2, # Threshold for block-level contamination decision.
    "TRESH_POLY": 0.2, # Threshold for polygon-level contamination decision.

}
...
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

```
Kendler, S. et al.
Hyperspectral imaging for chemicals identification:
a human-inspired machine learning approach
Scientific Reports, 2022
```

---

