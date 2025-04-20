# M5Forecast PatchTST (Ensemble-Based)

This repository provides a pretrained **PatchTST** model (and optionally multiple checkpoints) trained on the **M5 Forecasting** dataset.  
The core feature is **ensemble learning** over multiple checkpoints using a linear regression strategy to automatically learn optimal weights and boost prediction accuracy.

---

## 🔧 Features

- ✅ **Multi-checkpoint ensemble prediction** using linear regression
- ✅ Automatic loading of multiple checkpoints from a directory
- ✅ Support for synthetic or real M5-format time series input
- ✅ Full configurability via command-line interface
- ✅ Optional standalone inference with `inference.py`

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your_username/M5Forecast_PatchTST.git
cd M5Forecast_PatchTST
```

---

### 2. Install dependencies

We recommend using conda:

```bash
conda env create -f environment.yaml
conda activate m5patchtst
```

---

### 3. Run ensemble prediction (recommended)

This will:
- Load up to `N` checkpoints from `checkpoints_patchtst/`
- Use them to predict future time series values
- Learn a set of linear weights to combine the outputs
- Save the learned weights and final prediction as `.npy`

```bash
python ensemble_patchtst.py \
    --hist_len 366 \
    --pred_len 28 \
    --num_checkpoints 3 \
    --ckpt_dir ./checkpoints_patchtst/ \
    --config_path ./checkpoints_patchtst/args.yaml \
    --csv_path series.csv \
    --out_dir results
```

**Note:** If `series.csv` doesn't exist, a synthetic one will be automatically generated.

---

## 🧪 Optional: Single-model Inference (for testing)

You can also run inference on a single checkpoint using:

```bash
python inference.py
```

This will:
- Load `checkpoints/checkpoint.pth`
- Run a dummy input `[1, 366, 1]` through the model
- Print prediction shape `[1, 28, 1]`

---

## 📥 Input / Output Format

### Input:  
Shape = `[batch_size, hist_len=366, channels=1]`  
→ represents the historical input window

### Output:  
Shape = `[batch_size, pred_len=28, channels=1]`  
→ represents the future predictions

---

## M5 Forecasting Performance

The current implementation uses ensemble learning on top of a hierarchical modeling strategy to improve forecasting performance on the M5 dataset.

We collected a total of 220 checkpoints located in the `checkpoints_patchtst/` directory.  
Based on this ensemble method and additional post-processing, the WRMSSE evaluation results on the M5 test set are as follows:

| Level    | WRMSSE  |
|----------|---------|
| Avg      | 0.6210  |
| Level 1  | 0.3498  |
| Level 2  | 0.4552  |
| Level 3  | 0.5509  |
| Level 4  | 0.4028  |
| Level 5  | 0.5129  |
| Level 6  | 0.5363  |
| Level 7  | 0.6211  |
| Level 8  | 0.6238  |
| Level 9  | 0.7080  |
| Level 10 | 0.9260  |
| Level 11 | 0.8995  |
| Level 12 | 0.8654  |

These results demonstrate the effectiveness of PatchTST when combined with ensemble learning for hierarchical time series forecasting.

---

## 📄 License

MIT License. Feel free to use and cite.
