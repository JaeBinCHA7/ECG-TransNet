"""
Docstring for Options:
This class centralizes the argument parser for ECG classification.
It defines hyperparameters and settings used for data processing,
model architecture, and training configurations.
"""

import argparse


class Options:
    def __init__(self):
        pass

    def init(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        # ----------------------------
        # Global / training control
        # ----------------------------
        parser.add_argument('--batch_size', type=int, default=128,
                            help='Mini-batch size.')
        parser.add_argument('--nepoch', type=int, default=20,
                            help='Total number of training epochs.')
        parser.add_argument('--optimizer', type=str, default='adam',
                            help='Optimizer type (e.g., "adam", "sgd").')
        parser.add_argument('--lr_initial', type=float, default=5e-4,
                            help='Initial learning rate.')
        parser.add_argument('--gamma', type=float, default=0.1,
                            help='LR scheduler decay factor (used by MultiStepLR).')

        # ----------------------------
        # Logging / device
        # ----------------------------
        parser.add_argument('--log_name', type=str, default='Superclass',
                            help='Run name prefix for logging.')
        parser.add_argument('--device', type=str, default='cuda',
                            help='Compute device ("cuda" or "cpu").')

        # ----------------------------
        # Model / architecture
        # (added fields to match training script usage)
        # ----------------------------
        parser.add_argument('--arch', type=str, default='ECGTransNet',
                            help='Model architecture name (for logging).')
        parser.add_argument('--classes', type=int, default=5,
                            help='Number of output classes (e.g., 5=superclass, 23=subclass, 12=rhythm).')
        parser.add_argument('--head', type=int, default=16,
                            help='Number of attention heads.')
        parser.add_argument('--d_model', type=int, default=512,
                            help='Transformer embedding dimension.')
        parser.add_argument('--d_ff', type=int, default=8,
                            help='Transformer feed-forward dimension.')
        parser.add_argument('--num_layers', type=int, default=1,
                            help='Number of transformer layers.')
        parser.add_argument('--lead', type=int, default=12,
                            help='Number of ECG leads.')
        parser.add_argument('--drop_out', type=float, default=0.2,
                            help='Dropout rate.')

        # ----------------------------
        # Proxy-related (metric learning)
        # (names/usage align with training script/log_dir)
        # ----------------------------
        parser.add_argument('--proxy_weight', type=float, default=0.01,
                            help='Weight for proxy-related loss/branch.')
        parser.add_argument('--proxy_a', type=float, default=16,
                            help='Proxy hyperparameter (e.g., scale).')
        parser.add_argument('--proxy_d', type=float, default=0.1,
                            help='Proxy hyperparameter (e.g., margin).')

        # ----------------------------
        # Pretrained / resume
        # ----------------------------
        parser.add_argument('--pretrained', type=bool, default=False,
                            help='Set True to load from a pretrained checkpoint.')
        parser.add_argument('--pretrained_model', type=str,
                            # default='./log/-/models/best_model.pt',
                            default='./log/Subclass_20250922_160928_ECGTransNet_mode-train_loss-bce_bs128_lr0.0005_E512_H16_Proxy_w0.01_a16_d0.1/models/best_model.pt',
                            help='Path to pretrained model checkpoint.')

        # ----------------------------
        # Dataset / preprocessing
        # ----------------------------
        parser.add_argument('--fs', type=int, default=500,
                            help='Sampling frequency of ECG.')
        parser.add_argument('--label_type', type=str, default='label_diag_superclass',
                            help='PTB-XL label type (e.g., label_all, label_diag_superclass, '
                                 'label_diag_subclass, label_rhythm).')

        parser.add_argument('--data_length', type=int, default=10,
                            help='ECG segment length in seconds.')
        parser.add_argument('--samples', type=int, default=5000,
                            help='Number of samples to use from ECG data.')
        parser.add_argument('--data_path', type=str, default='./dataset/PTBXL_fs500',
                            help='Dataset root directory for training/validation.')

        return parser
