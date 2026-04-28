import os
import logging
import random
from pathlib import Path
from typing import List, Optional
import numpy as np
from torch.utils.data import DataLoader, Dataset

try:
    import pickle5 as pickle
except ImportError:
    import pickle


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def create_logger(name: str) -> logging.Logger:
    """
    Create a colored console logger. Avoids adding duplicate handlers
    if called multiple times.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()

    BEIGE, VIOLET, OKBLUE, ANTHRAZIT, ENDC = [
        '\033[32m', '\033[35m', '\033[94m', '\033[90m', '\033[0m'
    ]
    fmt = (BEIGE + '%(asctime)s ' + VIOLET + '%(name)s:' +
           OKBLUE + '%(lineno)s ' + ANTHRAZIT + '%(levelname)s: ' +
           ENDC + ' %(message)s')
    ch.setFormatter(logging.Formatter(fmt))
    logger.addHandler(ch)
    logger.propagate = False
    return logger


logger = create_logger(__name__)


# -----------------------------------------------------------------------------
# Dataset artifacts loader
# -----------------------------------------------------------------------------
def load_dataset(target_root: str, filename_postfix: str = "", df_mapped: bool = True):
    """
    Load dataset artifacts (df, lbl_itos, mean, std) produced during preprocessing.
    Mirrors original behavior (pickle5 compatibility, dtype handling).
    """
    target_root_path = Path(target_root)

    if df_mapped:
        df = pickle.load(open(target_root_path / ("df_memmap" + filename_postfix + ".pkl"), "rb"))
    else:
        df = pickle.load(open(target_root_path / ("df" + filename_postfix + ".pkl"), "rb"))

    lbl_pkl = target_root_path / ("lbl_itos" + filename_postfix + ".pkl")
    if lbl_pkl.exists():
        with open(lbl_pkl, "rb") as infile:
            lbl_itos = pickle.load(infile)
    else:
        lbl_itos = np.load(target_root_path / ("lbl_itos" + filename_postfix + ".npy"))

    mean = np.load(target_root_path / ("mean" + filename_postfix + ".npy"))
    std = np.load(target_root_path / ("std" + filename_postfix + ".npy"))
    return df, lbl_itos, mean, std


def multihot_encode(x: List[int], num_classes: int) -> np.ndarray:
    """Multi-hot encode a list/array of class indices of length C."""
    res = np.zeros(num_classes, dtype=np.float32)
    res[x] = 1
    return res


# -----------------------------------------------------------------------------
# Timeseries dataset with crop support
# -----------------------------------------------------------------------------
class TimeseriesDatasetCrops(Dataset):
    """Timeseries dataset that returns (possibly cropped) segments."""

    def __init__(self,
                 df,
                 output_size: int,
                 chunk_length: int,
                 min_chunk_length: int,
                 memmap_filename: Optional[str] = None,
                 npy_data=None,
                 random_crop: bool = True,
                 data_folder: Optional[str] = None,
                 num_classes: int = 2,
                 copies: int = 0,
                 col_lbl: Optional[str] = "label",
                 stride: Optional[int] = None,
                 start_idx: int = 0,
                 annotation: bool = False,
                 sample_items_per_record: int = 1):
        """
        Accepts three kinds of inputs:
        1) Filenames pointing to aligned npy arrays for data; labels can be ints/floats/arrays/filenames.
        2) A memmap path produced by reformat_as_memmap; df.data is an integer index into memmap.
        3) A single npy array [samples, ts, ...] or path; df.data is sample id.

        NOTE:
        - If using memmap or npy_data, df['data'] must be integer indices (np.int64).
        - col_lbl=None returns a dummy label 0 (e.g., for self-supervision).
        """
        assert not ((memmap_filename is not None) and (npy_data is not None)), \
            "Choose either memmap or npy_data, not both."
        # require integer data indices if using memmap or npy array
        assert (memmap_filename is None and npy_data is None) or df.data.dtype == np.int64

        self.output_size = output_size
        self.data_folder = Path(data_folder) if data_folder is not None else None
        self.annotation = annotation
        self.col_lbl = col_lbl
        self.c = num_classes

        # Store data column (bytes if filenames; int indices otherwise)
        self.timeseries_df_data = np.array(df["data"])
        if self.timeseries_df_data.dtype not in [np.int16, np.int32, np.int64]:
            # filenames mode: ensure bytes to avoid mp pickling issues
            assert memmap_filename is None and npy_data is None
            self.timeseries_df_data = np.array(df["data"].astype(str)).astype(np.string_)

        # Store labels
        if col_lbl is None:  # dummy labels
            self.timeseries_df_label = np.zeros(len(df))
        else:
            first = df[col_lbl].iloc[0]
            if isinstance(first, (list, np.ndarray)):
                # stack for proper batching
                self.timeseries_df_label = np.stack(df[col_lbl])
            else:
                self.timeseries_df_label = np.array(df[col_lbl])

            if self.timeseries_df_label.dtype not in [np.int16, np.int32, np.int64, np.float32, np.float64]:
                # filenames for annotations (string bytes)
                assert annotation and memmap_filename is None and npy_data is None
                self.timeseries_df_label = np.array(df[col_lbl].apply(lambda x: str(x))).astype(np.string_)

        # Operating mode
        self.mode = "files"
        if memmap_filename is not None:
            self.mode = "memmap"
            memmap_filename = Path(memmap_filename)
            self.memmap_meta_filename = memmap_filename.parent / (memmap_filename.stem + "_meta.npz")

            memmap_meta = np.load(self.memmap_meta_filename, allow_pickle=True)
            self.memmap_start = memmap_meta["start"]
            self.memmap_shape = memmap_meta["shape"]
            self.memmap_length = memmap_meta["length"]
            self.memmap_file_idx = memmap_meta["file_idx"]
            self.memmap_dtype = np.dtype(str(memmap_meta["dtype"]))
            # bytes to avoid mp issues
            self.memmap_filenames = np.array(memmap_meta["filenames"]).astype(str)

            if annotation:
                memmap_meta_label = np.load(
                    self.memmap_meta_filename.parent / (
                            "_".join(self.memmap_meta_filename.stem.split("_")[:-1]) + "_label_meta.npz"
                    ),
                    allow_pickle=True
                )
                self.memmap_shape_label = memmap_meta_label["shape"]
                self.memmap_filenames_label = np.array(memmap_meta_label["filenames"]).astype(str)
                self.memmap_dtype_label = np.dtype(str(memmap_meta_label["dtype"]))

        elif npy_data is not None:
            self.mode = "npy"
            if isinstance(npy_data, (np.ndarray, list)):
                self.npy_data = np.array(npy_data)
                assert annotation is False
            else:
                self.npy_data = np.load(npy_data, allow_pickle=True)
            if annotation:
                self.npy_data_label = np.load(Path(npy_data).parent / (Path(npy_data).stem + "_label.npy"),
                                              allow_pickle=True)

        self.random_crop = random_crop
        self.sample_items_per_record = sample_items_per_record

        # Build index mapping for crops
        self.df_idx_mapping: List[int] = []
        self.start_idx_mapping: List[int] = []
        self.end_idx_mapping: List[int] = []

        for df_idx, (_, row) in enumerate(df.iterrows()):
            if self.mode == "files":
                data_length = row["data_length"]
            elif self.mode == "memmap":
                data_length = self.memmap_length[row["data"]]
            else:  # npy mode
                data_length = len(self.npy_data[row["data"]])

            if chunk_length == 0:  # do not split
                idx_start = [start_idx]
                idx_end = [data_length]
            else:
                step = chunk_length if stride is None else stride
                idx_start = list(range(start_idx, data_length, step))
                idx_end = [min(s + chunk_length, data_length) for s in idx_start]

            # Remove trailing too-short chunk(s)
            for i in range(len(idx_start)):
                if (idx_end[i] - idx_start[i]) < min_chunk_length:
                    del idx_start[i:]
                    del idx_end[i:]
                    break

            for _ in range(copies + 1):
                for s, e in zip(idx_start, idx_end):
                    self.df_idx_mapping.append(df_idx)
                    self.start_idx_mapping.append(s)
                    self.end_idx_mapping.append(e)

        # Convert to ndarray to avoid mp issues
        self.df_idx_mapping = np.array(self.df_idx_mapping)
        self.start_idx_mapping = np.array(self.start_idx_mapping)
        self.end_idx_mapping = np.array(self.end_idx_mapping)

    def __len__(self) -> int:
        return len(self.df_idx_mapping)

    @property
    def is_empty(self) -> bool:
        return len(self.df_idx_mapping) == 0

    def __getitem__(self, idx: int):
        """
        Returns (data, label) OR a tuple of such pairs when sample_items_per_record > 1.
        """
        items = []
        for _ in range(self.sample_items_per_record):
            timesteps = self.get_sample_length(idx)

            if self.random_crop:
                if timesteps == self.output_size:
                    start_idx_rel = 0
                else:
                    # inclusive start, ensure at least output_size fits
                    start_idx_rel = random.randint(0, timesteps - self.output_size - 1)
            else:
                start_idx_rel = (timesteps - self.output_size) // 2

            if self.sample_items_per_record == 1:
                return self._getitem(idx, start_idx_rel)
            else:
                items.append(self._getitem(idx, start_idx_rel))
        return tuple(items)

    # Optional normalization utility (kept)
    def normalize_ecg(self, ecg_data: np.ndarray, axis=0) -> np.ndarray:
        """Z-score normalization for ECG."""
        mean = np.mean(ecg_data, axis=axis, keepdims=True)
        std = np.std(ecg_data, axis=axis, keepdims=True)
        return (ecg_data - mean) / (std + 1e-8)

    def _getitem(self, idx: int, start_idx_rel: int):
        """Low-level fetch of a single (data, label) pair with crop applied."""
        df_idx = self.df_idx_mapping[idx]
        start_idx = self.start_idx_mapping[idx]
        end_idx = self.end_idx_mapping[idx]

        timesteps = end_idx - start_idx
        assert timesteps >= self.output_size, "Crop window larger than available timesteps."

        start_idx_crop = start_idx + start_idx_rel
        end_idx_crop = start_idx_crop + self.output_size

        if self.mode == "files":
            data_filename = str(self.timeseries_df_data[df_idx])
            if self.data_folder is not None:
                data_filename = str(self.data_folder / data_filename)
            data = np.load(data_filename, allow_pickle=True)[start_idx_crop:end_idx_crop]

            if self.annotation:
                label_filename = str(self.timeseries_df_label[df_idx])
                if self.data_folder is not None:
                    label_filename = str(self.data_folder / label_filename)
                label = np.load(label_filename, allow_pickle=True)[start_idx_crop:end_idx_crop]
            else:
                label = self.timeseries_df_label[df_idx]

        elif self.mode == "memmap":
            memmap_idx = self.timeseries_df_data[df_idx]
            memmap_file_idx = self.memmap_file_idx[memmap_idx]
            idx_offset = self.memmap_start[memmap_idx]

            mem_filename = str(self.memmap_filenames[memmap_file_idx])
            mem_file = np.memmap(self.memmap_meta_filename.parent / mem_filename,
                                 dtype=self.memmap_dtype, mode='r',
                                 shape=tuple(self.memmap_shape[memmap_file_idx]))
            data = np.copy(mem_file[idx_offset + start_idx_crop: idx_offset + end_idx_crop])
            del mem_file

            if self.annotation:
                mem_filename_label = str(self.memmap_filenames_label[memmap_file_idx])
                mem_file_label = np.memmap(self.memmap_meta_filename.parent / mem_filename_label,
                                           dtype=self.memmap_dtype_label, mode='r',
                                           shape=tuple(self.memmap_shape_label[memmap_file_idx]))
                label = np.copy(mem_file_label[idx_offset + start_idx_crop: idx_offset + end_idx_crop])
                del mem_file_label
            else:
                label = self.timeseries_df_label[df_idx]

        else:  # "npy" mode
            ID = self.timeseries_df_data[df_idx]
            data = self.npy_data[ID][start_idx_crop:end_idx_crop]
            if self.annotation:
                label = self.npy_data_label[ID][start_idx_crop:end_idx_crop]
            else:
                label = self.timeseries_df_label[df_idx]

        return (data, label)

    # ---- helpers to introspect mapping ----
    def get_id_mapping(self) -> np.ndarray:
        return self.df_idx_mapping

    def get_sample_id(self, idx: int) -> int:
        return self.df_idx_mapping[idx]

    def get_sample_length(self, idx: int) -> int:
        return int(self.end_idx_mapping[idx] - self.start_idx_mapping[idx])

    def get_sample_start(self, idx: int) -> int:
        return int(self.start_idx_mapping[idx])


# -----------------------------------------------------------------------------
# Wrapper building train/val/test datasets & loaders
# -----------------------------------------------------------------------------
class SimCLRDataSetWrapper(object):
    """
    Wrapper to construct PTB-XL splits and dataloaders for a given label type.
    """

    def __init__(self,
                 batch_size: int,
                 num_workers: int,
                 target_folders: str,
                 target_fs: int,
                 ptb_xl_label: str = "label_diag_superclass",
                 folds: int = 8,
                 test: bool = False):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_folders = target_folders
        self.target_fs = target_fs
        self.ptb_xl_label = ptb_xl_label
        self.folds = folds
        self.test = test

        # Populated later
        self.val_ds_idmap = None
        self.lbl_itos = None
        self.num_classes = None
        self.train_ds_size = 0
        self.val_ds_size = 0
        self.df_train = None
        self.df_valid = None
        self.df_test = None

    def get_data_loaders(self):
        """Build datasets and return (train_loader, valid_loader, test_loader)."""
        train_ds, val_ds, test_ds = self._get_datasets(self.target_folders)
        self.val_ds_idmap = val_ds.get_id_mapping()

        train_loader, valid_loader, test_loader = self.get_train_validation_data_loaders(
            train_ds, val_ds, test_ds
        )

        self.train_ds_size = len(train_ds)
        self.val_ds_size = len(val_ds)
        return train_loader, valid_loader, test_loader

    def _get_datasets(self, target_folder: str, transforms=None):
        """
        Prepare PTB-XL splits (train/valid/test) and build TimeseriesDatasetCrops
        for each split, using memmap when available.
        """
        logger.info("get dataset from " + str(target_folder))

        # Fixed parameters (kept from original)
        input_size = 5000
        chunkify_train = False
        chunkify_valid = True
        chunk_length_train = input_size
        chunk_length_valid = input_size
        min_chunk_length = input_size
        stride_length_train = chunk_length_train // 4
        stride_length_valid = input_size // 2

        # Fold assignment
        if self.test:
            valid_fold = 10
            test_fold = 9
        else:
            valid_fold = 9
            test_fold = 10

        train_folds = list(range(1, 11))
        train_folds.remove(test_fold)
        train_folds.remove(valid_fold)
        train_folds = np.array(train_folds)

        # Artifact file names
        df_memmap_filename = "df_memmap.pkl"
        memmap_filename = "memmap.npy"

        # Load artifacts
        df_mapped, lbl_itos, mean, std = load_dataset(target_folder)
        # Overwrite df_mapped with explicit df_memmap.pkl (kept for exact compatibility)
        df_mapped = pickle.load(open(os.path.join(target_folder, df_memmap_filename), "rb"))

        self.lbl_itos = lbl_itos
        self.num_classes = len(lbl_itos)

        # Label column selection (PTB-XL specific)
        label_key = self.ptb_xl_label
        self.lbl_itos = np.array(lbl_itos[label_key])
        numeric_col = label_key + "_filtered_numeric"

        # For convenience
        df_mapped["diag_label"] = df_mapped[numeric_col].copy()
        logger.debug("get labels for linear evaluation on ptb")
        df_mapped["label"] = df_mapped[numeric_col].apply(
            lambda x: multihot_encode(x, len(self.lbl_itos))
        )

        # Split by folds; ensure positive labels
        assert (self.folds < 9)
        df_train = df_mapped[(df_mapped.strat_fold.apply(lambda x: x in train_folds[range(self.folds)]) &
                              (df_mapped.label.apply(lambda x: np.sum(x) > 0)))]
        df_valid = df_mapped[(df_mapped.strat_fold == valid_fold) &
                             (df_mapped.label.apply(lambda x: np.sum(x) > 0))]
        df_test = df_mapped[(df_mapped.strat_fold == test_fold) &
                            (df_mapped.label.apply(lambda x: np.sum(x) > 0))]

        # Build datasets (memmap-backed)
        train_ds = TimeseriesDatasetCrops(
            df_train, input_size, num_classes=len(self.lbl_itos),
            data_folder=target_folder,
            chunk_length=chunk_length_train if chunkify_train else 0,
            min_chunk_length=min_chunk_length, stride=stride_length_train,
            annotation=False, col_lbl="label",
            memmap_filename=os.path.join(target_folder, memmap_filename)
        )
        val_ds = TimeseriesDatasetCrops(
            df_valid, input_size, num_classes=len(self.lbl_itos),
            data_folder=target_folder,
            chunk_length=chunk_length_valid if chunkify_valid else 0,
            min_chunk_length=min_chunk_length, stride=stride_length_valid,
            annotation=False, col_lbl="label",
            memmap_filename=os.path.join(target_folder, memmap_filename)
        )
        test_ds = TimeseriesDatasetCrops(
            df_test, input_size, num_classes=len(self.lbl_itos),
            data_folder=target_folder,
            chunk_length=chunk_length_valid if chunkify_valid else 0,
            min_chunk_length=min_chunk_length, stride=stride_length_valid,
            annotation=False, col_lbl="label",
            memmap_filename=os.path.join(target_folder, memmap_filename)
        )

        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test = df_test
        return train_ds, val_ds, test_ds

    def get_train_validation_data_loaders(self, train_ds: Dataset, val_ds: Dataset, test_ds: Dataset):
        """
        Construct torch DataLoaders with the same parameters as the original code.
        """
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=True, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_ds, batch_size=1,
            shuffle=False, num_workers=self.num_workers, pin_memory=True
        )
        return train_loader, val_loader, test_loader


# -----------------------------------------------------------------------------
# Lightning-style DataModule (without Lightning)
# -----------------------------------------------------------------------------
class ECGDataModule:
    """
    Small helper to prepare datasets and provide train/valid/test DataLoaders.
    Mirrors the original behavior (batch sizes, shuffles, pin_memory).
    """

    def __init__(self,
                 opt,
                 num_workers: int = 8,
                 seed: int = 42):
        self.num_workers = num_workers
        self.batch_size = opt.batch_size
        self.data_path = opt.data_path
        self.fs = opt.fs
        self.label_type = opt.label_type
        self.seed = seed
        self.opt = opt

        self._prepare_data()

    def _prepare_data(self):
        """Prepare datasets and DataLoaders (single call version; semantics preserved)."""
        dataset = SimCLRDataSetWrapper(
            self.batch_size,
            self.num_workers,
            self.data_path,
            self.fs,
            self.label_type,
        )

        # Single call to build loaders; preserves outcomes while avoiding redundant work
        train_loader, valid_loader, test_loader = dataset.get_data_loaders()

        self.train_dataset = train_loader.dataset
        self.valid_dataset = valid_loader.dataset
        self.test_dataset = test_loader.dataset

        self.num_samples = dataset.train_ds_size

        # Mirror DataLoader settings used above (kept as in original)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    @property
    def num_classes(self) -> int:
        """Return number of classes (fixed 5 in original code)."""
        return 5

    def get_train_loader(self) -> DataLoader:
        return self.train_loader

    def get_valid_loader(self) -> DataLoader:
        return self.valid_loader

    def get_test_loader(self) -> DataLoader:
        return self.test_loader
