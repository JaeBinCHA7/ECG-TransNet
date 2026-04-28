# ECG TransNet: Intra/Inter-Lead Feature Integration with Proxy-Guided Learning for 12-Lead ECG Classification

[![Paper](https://img.shields.io/badge/IEEE_Xplore-Paper_Link-blue.svg)](https://ieeexplore.ieee.org/document/11460795)

> Official PyTorch implementation of "ECG TransNet: Intra/Inter-Lead Feature Integration with Proxy-Guided Learning for 12-Lead ECG Classification" (ICASSP 2026). A SOTA multi-label 12-lead ECG classification model on PTB-XL.

**📄 [Read the full paper on IEEE Xplore here](https://ieeexplore.ieee.org/document/11460795)**

## 📌 Abstract
**ECG TransNet** addresses multi-label classification of 12-lead ECGs by modeling both within-lead morphology and cross-lead interactions. Our architecture utilizes:
* **ASPP-Conv Block:** Extracts multi-scale temporal features for each lead.
* **Inter-lead Attention (ILA):** Captures cross-lead dependencies.
* **Head-independent Transformer (Hi-Transformer):** Refines representations without early mixing across heads.
* **Proxy-Guided Learning:** Sharpens per-head decision boundaries and aligns class proxies to handle multi-label and long-tailed ECG distributions.

---

## 🚀 Getting Started

Follow the steps below to replicate the environment, prepare the dataset, and train the model.

### 1. Requirements & Installation
First, clone the repository and install the required dependencies. It is recommended to use a virtual environment (e.g., Anaconda or venv).

```bash
git clone [https://github.com/your-username/ECG-TransNet.git](https://github.com/your-username/ECG-TransNet.git)
cd ECG-TransNet
pip install -r requirments.txt
```

### 2. Dataset Preparation (PTB-XL)
Download the **PTB-XL dataset (version 1.0.3)**, a large publicly available electrocardiography dataset.
* **Download Link:** [PTB-XL on PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/)
* Extract the downloaded ZIP file into a directory of your choice.

### 3. Data Preprocessing
We provide a preprocessing script to resample the raw `.dat` and `.hea` files and convert them into efficient Memory-Mapped (`memmap.npy`) formats.

Open `data_preprocess.py` and modify the `data_root` to point to your downloaded PTB-XL directory:

```python
# Inside data_preprocess.py (Line ~336)
if __name__ == '__main__':
    target_fs = 500
    data_root = Path('/path/to/your/ptb-xl-dataset/') # <-- Change this path
    target_root = Path("./dataset")
```

Then run the script:

```bash
python data_preprocess.py
```

This will create a `dataset/PTBXL_fs500` folder containing the preprocessed data artifacts.

### 4. Configuration (`options.py`)
Model hyperparameters, dataset paths, and training configurations are centralized in `options.py`. 
You can modify the default values directly in the file, or pass them as arguments via the command line.

**Key Arguments:**
* `--data_path`: Path to the preprocessed dataset (default: `./dataset/PTBXL_fs500`).
* `--label_type`: Target task. Choose from `label_diag_superclass` (5 classes), `label_diag_subclass` (23 classes), or `label_rhythm` (12 classes).
* `--log_name`: Prefix for your logging directory.
* `--batch_size`, `--lr_initial`, `--proxy_weight`, `--head`, etc.

### 5. Training the Model
To start training ECG TransNet from scratch, run `train.py`. The script will automatically log the training process, metrics, and save the best model weights inside the `./log/` directory.

Example for training the **Superclass** task:

```bash
python train.py --log_name Superclass_Run --label_type label_diag_superclass --batch_size 128 --lr_initial 0.0005
```

Example for training the **Subclass** task:

```bash
python train.py --log_name Subclass_Run --label_type label_diag_subclass --classes 23
```

### 6. Evaluation (Testing)
To evaluate a trained model, run `test.py` and point `--pretrained_model` to your saved `.pt` checkpoint. It will calculate AUROC, AUPRC, F1-score, and save a consolidated report.

```bash
python test.py --pretrained True --pretrained_model ./log/Your_Run_Folder/models/best_model.pt
```

---

## 📝 Citation
If you find this code or our paper useful in your research, please cite our work:

```bibtex
@inproceedings{cha2026ecg,
  title={ECG TransNet: Intra/Inter-Lead Feature Integration with Proxy-Guided Learning for 12-Lead ECG Classification},
  author={Cha, Jaebin and Heo, Junyeong and Kim, Ryuha and Cho, Sungpil and Park, Youngcheol},
  booktitle={ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={8017--8021},
  year={2026},
  organization={IEEE}
}
```

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
