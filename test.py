"""
Test interface for ECG classification (cleaned).
- Uses only scikit-learn for all metrics.
- Prints a single consolidated report and saves it to one text file.
"""
import argparse
import os
import random
import numpy as np
import torch
from colorama import Fore, Style, init
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix,
    cohen_kappa_score,
    hamming_loss,
    accuracy_score,
    precision_recall_curve,
)

import options
import utils
from dataloader import ECGDataModule
from model import ECGTransNet

# ----------------------------
# UI
# ----------------------------
init(autoreset=True)
DECOR = Fore.YELLOW + "#" * 68

# ----------------------------
# Helpers (metrics)
# ----------------------------
def best_thresholds_by_f1(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """
    Per-class threshold that maximizes F1 using PR curve points.
    - Safe F1 calculation (masking to avoid 0/0)
    - Falls back to 0.5 when a class lacks positives/negatives or thresholds are empty
    """
    from sklearn.metrics import precision_recall_curve

    C = y_true.shape[1]
    thr = np.full(C, 0.5, dtype=np.float32)  # default 0.5

    for c in range(C):
        # need both positive and negative samples
        if len(np.unique(y_true[:, c])) < 2:
            continue

        precision, recall, thresholds = precision_recall_curve(y_true[:, c], y_prob[:, c])

        if thresholds.size == 0:
            continue

        # align lengths: last point has no threshold
        p = precision[:-1]
        r = recall[:-1]

        denom = p + r
        mask = denom > 0

        f1 = np.zeros_like(denom, dtype=np.float64)
        if np.any(mask):
            f1[mask] = (2.0 * p[mask] * r[mask]) / denom[mask]
            thr[c] = thresholds[int(np.nanargmax(f1))]
        else:
            thr[c] = thresholds[int(np.argmax(p + r))]

    return thr


def binarize_with_thresholds(y_prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Apply per-class thresholds to probability matrix."""
    return (y_prob >= thresholds.reshape(1, -1)).astype(np.int32)


def per_class_specificity(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> np.ndarray:
    """
    Specificity = TN / (TN + FP) per class.
    """
    C = y_true_bin.shape[1]
    spec = np.zeros(C, dtype=np.float64)
    for c in range(C):
        cm = confusion_matrix(y_true_bin[:, c], y_pred_bin[:, c], labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            spec[c] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            spec[c] = 0.0
    return spec


def mean_cohen_kappa(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> float:
    """
    Average Cohen's kappa across classes.
    """
    kappas = []
    C = y_true_bin.shape[1]
    for c in range(C):
        if len(np.unique(y_true_bin[:, c])) < 2 and len(np.unique(y_pred_bin[:, c])) < 2:
            continue
        kappas.append(cohen_kappa_score(y_true_bin[:, c], y_pred_bin[:, c]))
    return float(np.mean(kappas)) if len(kappas) > 0 else 0.0


# ----------------------------
# Parser / Options
# ----------------------------
opt = options.Options().init(argparse.ArgumentParser(description='ECG Classification')).parse_args()
print(opt)

# ----------------------------
# Device / Seeds
# ----------------------------
DEVICE = torch.device(opt.device)
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

# ----------------------------
# Model
# ----------------------------
model = ECGTransNet(opt=opt)

total_params = utils.cal_total_params(model)
print('total params   : %d (%.2f M, %.2f MBytes)\n' %
      (total_params, total_params / 1e6, total_params * 4.0 / 1e6))

print('Load the pretrained model...')
chkpt = torch.load(opt.pretrained_model, weights_only=False)
model.load_state_dict(chkpt['model'])
model = model.to(DEVICE)

# ----------------------------
# Data
# ----------------------------
data_module = ECGDataModule(opt)
test_loader = data_module.get_test_loader()

# ----------------------------
# Inference
# ----------------------------
print('Validation start...')
y_prob_list, y_true_list = [], []
model.eval()
with torch.no_grad():
    for inputs, targets in utils.Bar(test_loader):
        inputs = inputs.float().to(DEVICE).permute(0, 2, 1)  # [B, T, C] -> model expects [B, C, T]
        targets = targets.float().to(DEVICE)

        logits, _, _ = model(inputs)  # [B, C]
        probs = torch.sigmoid(logits)

        y_prob_list.append(probs.detach().cpu().numpy())
        y_true_list.append(targets.detach().cpu().numpy())

y_prob = np.concatenate(y_prob_list, axis=0)  # (N, C)
y_true = np.concatenate(y_true_list, axis=0)  # (N, C)

# ----------------------------
# Metrics (threshold-free)
# ----------------------------
# AUROC/AUPRC (macro & per-class)
try:
    auroc_macro = roc_auc_score(y_true=y_true, y_score=y_prob, average='macro')
    auroc_per_class = roc_auc_score(y_true=y_true, y_score=y_prob, average=None)
except ValueError:
    # Some classes might have a single label present
    auroc_macro = float('nan')
    auroc_per_class = np.full(y_true.shape[1], np.nan, dtype=np.float64)

auprc_macro = average_precision_score(y_true=y_true, y_score=y_prob, average='macro')
auprc_per_class = average_precision_score(y_true=y_true, y_score=y_prob, average=None)

# ----------------------------
# Thresholding
# ----------------------------
thr_best = best_thresholds_by_f1(y_true, y_prob)
thr_fixed = np.full(y_true.shape[1], 0.5, dtype=np.float32)

y_pred_best = binarize_with_thresholds(y_prob, thr_best)
y_pred_05 = binarize_with_thresholds(y_prob, thr_fixed)

# F1/Precision/Recall @ best-thr (per-class & macro)
prec_best_pc, rec_best_pc, f1_best_pc, _ = precision_recall_fscore_support(
    y_true, y_pred_best, average=None, zero_division=0
)
prec_best = float(np.mean(prec_best_pc))
rec_best = float(np.mean(rec_best_pc))
f1_best = float(np.mean(f1_best_pc))

# Specificity @ best-thr
spec_best_pc = per_class_specificity(y_true, y_pred_best)
spec_best = float(np.mean(spec_best_pc))

# F1 @ 0.5 (macro & per-class)
_, _, f1_05_pc, _ = precision_recall_fscore_support(
    y_true, y_pred_05, average=None, zero_division=0
)
f1_05 = float(np.mean(f1_05_pc))

# Kappa (class-wise average) & Hamming/Subset accuracy
kappa = mean_cohen_kappa(y_true, y_pred_best)
hamm = hamming_loss(y_true, y_pred_best)
subset_acc = accuracy_score(y_true, y_pred_best)  # exact match ratio (subset accuracy)

# ---- Macro accuracy (per-class accuracy mean) ----
acc_per_class = (y_true == y_pred_best).mean(axis=0)      # per-class accuracy
macro_acc = float(acc_per_class.mean())                   # macro accuracy

# ----------------------------
# Report (print + save once)
# ----------------------------
save_tag = opt.pretrained_model.split('/')[2] if '/' in opt.pretrained_model else os.path.basename(opt.pretrained_model)
log_dir = os.path.join('./log', save_tag)
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'{save_tag}_test_summary.txt')

def fmt_arr(arr, digits=4):
    return np.array2string(np.round(arr.astype(np.float64), digits), separator=', ')

print(DECOR)
print(Fore.CYAN + Style.BRIGHT + "# Test score (Consolidated)")
print(Fore.MAGENTA + str(opt.pretrained_model))
print(Fore.GREEN + 'total params   : %d (%.2f M, %.2f MBytes)' %
      (total_params, total_params / 1e6, total_params * 4.0 / 1e6))
print(Fore.BLUE + 'Best Thresholds: ' + Fore.YELLOW + str(np.round(thr_best, 4)))
print(Fore.RED + f"AUROC (macro) : {auroc_macro:.4f}")
print(Fore.RED + f"AUPRC (macro) : {auprc_macro:.4f}")
print(Fore.RED + f"F1 (best-thr) : {f1_best:.4f} | Precision: {prec_best:.4f} | Recall: {rec_best:.4f} | Specificity: {spec_best:.4f}")
print(Fore.RED + f"F1 (@0.5)     : {f1_05:.4f}")
print(Fore.RED + f"Cohen's kappa : {kappa:.4f}")
print(Fore.RED + f"Hamming loss  : {hamm:.4f}")
print(Fore.RED + f"Subset acc    : {subset_acc:.4f}")
print(Fore.RED + f"Macro acc     : {macro_acc:.4f}")
print(Fore.CYAN + "# Per-class AUROC : " + fmt_arr(auroc_per_class))
print(Fore.CYAN + "# Per-class AUPRC : " + fmt_arr(auprc_per_class))
print(Fore.CYAN + "# Per-class F1(best-thr)     : " + fmt_arr(f1_best_pc))
print(Fore.CYAN + "# Per-class Spec(best-thr)   : " + fmt_arr(spec_best_pc))
print(Fore.CYAN + "# Per-class Accuracy(best-thr): " + fmt_arr(acc_per_class))
print(DECOR)

with open(log_file, 'w', encoding='utf-8') as f:
    f.write("#" * 68 + "\n")
    f.write("# Test score (Consolidated)\n")
    f.write(f"{str(opt.pretrained_model)}\n")
    f.write('total params   : %d (%.2f M, %.2f MBytes)\n\n' %
            (total_params, total_params / 1e6, total_params * 4.0 / 1e6))
    f.write(f'Best Thresholds: {np.array2string(np.round(thr_best, 6), separator=", ")}\n')
    f.write(f"AUROC (macro) : {auroc_macro:.6f}\n")
    f.write(f"AUPRC (macro) : {auprc_macro:.6f}\n")
    f.write(f"F1 (best-thr) : {f1_best:.6f}\n")
    f.write(f"Precision (best-thr) : {prec_best:.6f}\n")
    f.write(f"Recall (best-thr)    : {rec_best:.6f}\n")
    f.write(f"Specificity (best-thr): {spec_best:.6f}\n")
    f.write(f"F1 (@0.5)     : {f1_05:.6f}\n")
    f.write(f"Cohen_kappa   : {kappa:.6f}\n")
    f.write(f"Hamming_loss  : {hamm:.6f}\n")
    f.write(f"Subset_acc    : {subset_acc:.6f}\n")
    f.write(f"Macro_acc     : {macro_acc:.6f}\n\n")

    f.write("# Per-class AUROC:\n")
    f.write(f"{fmt_arr(auroc_per_class, 6)}\n")
    f.write("# Per-class AUPRC:\n")
    f.write(f"{fmt_arr(auprc_per_class, 6)}\n")
    f.write("# Per-class F1 (best-thr):\n")
    f.write(f"{fmt_arr(f1_best_pc, 6)}\n")
    f.write("# Per-class Specificity (best-thr):\n")
    f.write(f"{fmt_arr(spec_best_pc, 6)}\n")
    f.write("# Per-class Accuracy (best-thr):\n")
    f.write(f"{fmt_arr(acc_per_class, 6)}\n")

    f.write("#" * 68 + "\n")
    f.write('System has been finished.\n')
