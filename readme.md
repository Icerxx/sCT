# Project Title

A deep learning project for CBCT2CT.

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Icerxx/sCT.git
cd sCT
```

### 2. Create environment
```bash
conda create -n sct python=3.10
conda activate sct

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Assets
- Pre-trained Weights: Download the [`model.pth`](https://drive.google.com/file/d/1_MgctUnIIFcQJCVOhkcs84qq92hlqXVA/view) and place it in the **root directory**.

- Dataset: Download the [synthRAD2025_Task2_Train.zip](https://zenodo.org/records/15373853). Extract and place the files inside the `data` directory.

Your folder should look like this:
```
.
├── data/
│   ├── [synthRAD2025_Task2_Train]
│   └── prepare_data.py
├── model.pth            <-- Place downloaded weights
├── train.py
├── inference.py
├── evaluate.py
└── requirements.txt
```

### 4. Prepare Data
Convert the data to .npy and divide into train, validation, and test sets.
```bash
cd data
python prepare_data.py
```

## 🚀 Usage
### Step1: Train the Model
> Note: Return to the **root directory** before proceeding to the next steps.

```bash
cd ..
python train.py
```

### Step2: Run Inference
```bash
python inference.py
```

### Step3: Evaluate the Model
```bash
python evaluate.py
```