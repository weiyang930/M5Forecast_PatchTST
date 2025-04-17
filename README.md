# PatchTST for M5 Forecasting

This repository provides a pretrained PatchTST model trained on the M5 Forecasting dataset. It supports direct inference using `inference.py`.

## Usage

### 1. Clone the repository and enter the folder

```bash
git clone https://github.com/your_username/m5_patchtst.git
cd m5_patchtst
```

### 2. Install dependencies

```bash
conda env create -f environment.yaml
```


### 3. Run inference

```bash
python inference.py
```

This will:

- Load pretrained weights from `checkpoints/checkpoint.pth`
- Load model configuration from `checkpoints/args.yaml`
- Run a forward pass using dummy input `[1, 512, 1]`
- Print the prediction shape

### 4. Directory structure

```
.
├── model.py
├── inference.py
├── checkpoints/
│     ├── checkpoint.pth
│     └── args.yaml
├── requirements.txt
└── README.md
```

### 5. Input Format

Input tensor shape should be:

```
[batch_size, sequence_length=366, num_channel=1]
```

Output will be:

```
[batch_size, prediction_length=28, num_channel=1]
```