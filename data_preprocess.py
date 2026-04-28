from pathlib import Path
from typing import Dict, List, Tuple, Union, Iterable

import numpy as np
import pandas as pd
import wfdb
from skimage import transform
from scipy.ndimage import zoom
from tqdm.auto import tqdm

try:
    import pickle5 as pickle
except ImportError:
    import pickle

# ---------------------------------------------------------------------
# Defaults / constants
# ---------------------------------------------------------------------

channel_stoi_default: Dict[str, int] = {
    "i": 0, "ii": 1, "iii": 2,
    "avr": 3, "avl": 4, "avf": 5,
    "v1": 6, "v2": 7, "v3": 8,
    "v4": 9, "v5": 10, "v6": 11
}


# ---------------------------------------------------------------------
# Memmap helpers
# ---------------------------------------------------------------------

def npys_to_memmap(npys: Iterable[Path],
                   target_filename: Path,
                   max_len: int = 0,
                   delete_npys: bool = True) -> None:
    """
    Concatenate multiple .npy arrays into a (possibly sharded) memmap file.
    Saves a companion *_meta.npz with indexing information.

    Behavior:
    - If max_len == 0, everything is written into target_filename.
    - If max_len  > 0, shards target_filename_{k}.npy will be created.

    This function keeps the original behavior/semantics.
    """
    memmap = None
    start: List[int] = []  # start index in current memmap file for each sample
    length: List[int] = []  # length of each sample
    filenames: List[Path] = []  # list of memmap file paths
    file_idx: List[int] = []  # memmap-file index per sample
    shape: List[List[int]] = []

    npys = list(npys)

    for idx, npy in tqdm(list(enumerate(npys))):
        data = np.load(npy, allow_pickle=True)

        # decide whether to create a new shard
        create_new = (memmap is None) or (max_len > 0 and (start[-1] + length[-1] > max_len))
        if create_new:
            if max_len > 0:
                # fix a small bug in original string concatenation for filenames
                filenames.append(target_filename.parent / f"{target_filename.stem}_{len(filenames)}.npy")
            else:
                filenames.append(target_filename)

            # if previous memmap existed and exceeded, record its final shape then close
            if memmap is not None:
                shape.append([start[-1] + length[-1]] + [l for l in data.shape[1:]])
                del memmap

            # initialize first segment in new shard
            start.append(0)
            length.append(int(data.shape[0]))
            memmap = np.memmap(filenames[-1], dtype=data.dtype, mode='w+', shape=data.shape)

        else:
            # append to existing shard: extend memmap shape and write
            start.append(start[-1] + length[-1])
            length.append(int(data.shape[0]))
            new_shape = tuple([start[-1] + length[-1]] + [l for l in data.shape[1:]])
            memmap = np.memmap(filenames[-1], dtype=data.dtype, mode='r+', shape=new_shape)

        # mapping from sample -> shard id
        file_idx.append(len(filenames) - 1)

        # actual write
        memmap[start[-1]: start[-1] + length[-1]] = data[:]
        memmap.flush()

        if delete_npys:
            try:
                npy.unlink()
            except Exception:
                pass

    # close last shard
    if memmap is not None:
        del memmap

    # append final shape for the last shard if needed
    if len(shape) < len(filenames):
        # use last loaded 'data' shape (same as original code intent)
        shape.append([start[-1] + length[-1]] + [l for l in data.shape[1:]])  # noqa: F821 (data defined in loop)

    # save meta info (filenames saved as relative names per original behavior)
    filenames_rel = [f.name for f in filenames]
    np.savez(
        target_filename.parent / f"{target_filename.stem}_meta.npz",
        start=np.array(start),
        length=np.array(length),
        shape=np.array(shape, dtype=object),
        file_idx=np.array(file_idx),
        dtype=data.dtype,  # noqa: F821
        filenames=np.array(filenames_rel, dtype=object)
    )


def npys_to_memmap_batched(npys: Iterable[Path],
                           target_filename: Path,
                           max_len: int = 0,
                           delete_npys: bool = True,
                           batch_length: int = 900000) -> None:
    """
    Batched version of npys_to_memmap to reduce memory overhead.
    Concatenates incrementally when accumulated length exceeds 'batch_length'.
    Produces the same meta artifacts as npys_to_memmap.
    """
    memmap = None
    start = np.array([0], dtype=np.int64)  # next start positions (last elem trimmed at the end)
    length: np.ndarray = np.array([], dtype=np.int64)
    filenames: List[Path] = []
    file_idx: np.ndarray = np.array([], dtype=np.int64)
    shape: List[List[int]] = []
    dtype = None

    npys = list(npys)
    data_buf: List[np.ndarray] = []
    data_lengths: List[int] = []

    for idx, npy in tqdm(list(enumerate(npys))):
        arr = np.load(npy, allow_pickle=True)
        data_buf.append(arr)
        data_lengths.append(len(arr))

        flush_now = (idx == len(npys) - 1) or (np.sum(data_lengths) > batch_length)
        if flush_now:
            data = np.concatenate(data_buf)
            data_total_len = int(np.sum(data_lengths))

            # open or rotate shard if necessary
            new_shard = (memmap is None) or (max_len > 0 and start[-1] > max_len)
            if new_shard:
                if max_len > 0:
                    filenames.append(target_filename.parent / f"{target_filename.stem}_{len(filenames)}.npy")
                else:
                    filenames.append(target_filename)

                shape.append([data_total_len] + [l for l in data.shape[1:]])
                if memmap is not None:
                    del memmap

                start[-1] = 0
                start = np.concatenate([start, np.cumsum(data_lengths, dtype=np.int64)])
                length = np.concatenate([length, np.array(data_lengths, dtype=np.int64)])

                memmap = np.memmap(filenames[-1], dtype=data.dtype, mode='w+', shape=data.shape)

            else:
                # append to existing shard
                start = np.concatenate([start, start[-1] + np.cumsum(data_lengths, dtype=np.int64)])
                length = np.concatenate([length, np.array(data_lengths, dtype=np.int64)])
                shape[-1] = [int(start[-1])] + [l for l in data.shape[1:]]
                memmap = np.memmap(filenames[-1], dtype=data.dtype, mode='r+', shape=tuple(shape[-1]))

            # map each sample to current shard
            file_idx = np.concatenate([file_idx, np.full(len(data_lengths), len(filenames) - 1, dtype=np.int64)])

            # write and flush
            write_start = int(start[-len(data_lengths) - 1])
            memmap[write_start: write_start + len(data)] = data[:]
            memmap.flush()

            dtype = data.dtype
            data_buf = []
            data_lengths = []

    # trim last 'start' helper
    start = start[:-1]

    # cleanup .npy fragments
    if delete_npys:
        for npy in npys:
            try:
                Path(npy).unlink()
            except Exception:
                pass

    if memmap is not None:
        del memmap

    filenames_rel = [f.name for f in filenames]
    np.savez(
        target_filename.parent / f"{target_filename.stem}_meta.npz",
        start=start,
        length=length,
        shape=np.array(shape, dtype=object),
        file_idx=file_idx,
        dtype=dtype,
        filenames=np.array(filenames_rel, dtype=object)
    )


def reformat_as_memmap(df: pd.DataFrame,
                       target_filename: Path,
                       data_folder: Union[None, Path] = None,
                       annotation: bool = False,
                       max_len: int = 0,
                       delete_npys: bool = True,
                       col_data: str = "data",
                       col_label: str = "label",
                       batch_length: int = 0) -> pd.DataFrame:
    """
    Convert per-record .npy files referenced by df[col_data] (and optionally labels) into a memmap dataset.
    Returns a new dataframe where 'data' column is replaced by integer indices mapping into memmap.
    """
    npys_data: List[Path] = []
    npys_label: List[Path] = []

    for _, row in df.iterrows():
        npys_data.append((data_folder / row[col_data]) if data_folder is not None else Path(row[col_data]))
        if annotation:
            npys_label.append((data_folder / row[col_label]) if data_folder is not None else Path(row[col_label]))

    if batch_length == 0:
        npys_to_memmap(npys_data, target_filename, max_len=max_len, delete_npys=delete_npys)
    else:
        npys_to_memmap_batched(
            npys_data, target_filename, max_len=max_len, delete_npys=delete_npys, batch_length=batch_length
        )

    if annotation:
        label_target = target_filename.parent / f"{target_filename.stem}_label.npy"
        if batch_length == 0:
            npys_to_memmap(npys_label, label_target, max_len=max_len, delete_npys=delete_npys)
        else:
            npys_to_memmap_batched(
                npys_label, label_target, max_len=max_len, delete_npys=delete_npys, batch_length=batch_length
            )

    # map data filename -> integer index
    df_mapped = df.copy()
    df_mapped["data_original"] = df_mapped[col_data]
    df_mapped[col_data] = np.arange(len(df_mapped))
    df_mapped.to_pickle(target_filename.parent / f"df_{target_filename.stem}.pkl")
    return df_mapped


# ---------------------------------------------------------------------
# Dataset save/load utilities
# ---------------------------------------------------------------------

def save_dataset(df: pd.DataFrame,
                 lbl_itos: Union[Dict, np.ndarray],
                 mean: np.ndarray,
                 std: np.ndarray,
                 target_root: Union[str, Path],
                 filename_postfix: str = "",
                 protocol: int = 4) -> None:
    """
    Persist dataset artifacts (df, label map, mean, std) under target_root.
    """
    target_root = Path(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    df.to_pickle(target_root / f"df{filename_postfix}.pkl", protocol=protocol)

    if isinstance(lbl_itos, dict):
        with open(target_root / f"lbl_itos{filename_postfix}.pkl", "wb") as f:
            pickle.dump(lbl_itos, f, protocol=protocol)
    else:
        np.save(target_root / f"lbl_itos{filename_postfix}.npy", lbl_itos)

    np.save(target_root / f"mean{filename_postfix}.npy", mean)
    np.save(target_root / f"std{filename_postfix}.npy", std)


def load_dataset(target_root: Union[str, Path],
                 filename_postfix: str = "",
                 df_mapped: bool = True) -> Tuple[pd.DataFrame, Union[Dict, np.ndarray], np.ndarray, np.ndarray]:
    """
    Load dataset artifacts saved by save_dataset / reformat.
    """
    target_root = Path(target_root)

    df_path = target_root / (f"df_memmap{filename_postfix}.pkl" if df_mapped
                             else f"df{filename_postfix}.pkl")
    df = pickle.load(open(df_path, "rb"))

    lbl_pkl = target_root / f"lbl_itos{filename_postfix}.pkl"
    if lbl_pkl.exists():
        with open(lbl_pkl, "rb") as f:
            lbl_itos = pickle.load(f)
    else:
        lbl_itos = np.load(target_root / f"lbl_itos{filename_postfix}.npy", allow_pickle=True)

    mean = np.load(target_root / f"mean{filename_postfix}.npy")
    std = np.load(target_root / f"std{filename_postfix}.npy")
    return df, lbl_itos, mean, std


# ---------------------------------------------------------------------
# Dataset stats helpers
# ---------------------------------------------------------------------

def dataset_add_length_col(df, col="data", data_folder=None):
    '''add a length column to the dataset df'''
    df[col + "_length"] = df[col].apply(
        lambda x: len(np.load(x if data_folder is None else data_folder / x, allow_pickle=True)))


def dataset_add_mean_col(df, col="data", axis=(0), data_folder=None):
    '''adds a column with mean'''
    df[col + "_mean"] = df[col].apply(
        lambda x: np.mean(np.load(x if data_folder is None else data_folder / x, allow_pickle=True), axis=axis))


def dataset_add_std_col(df, col="data", axis=(0), data_folder=None):
    '''adds a column with mean'''
    df[col + "_std"] = df[col].apply(
        lambda x: np.std(np.load(x if data_folder is None else data_folder / x, allow_pickle=True), axis=axis))


def dataset_get_stats(df: pd.DataFrame,
                      col: str = "data",
                      simple: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate mean/std across dataset.
    - simple=True: unweighted mean of per-file means/stds.
    - simple=False: pooled (weighted) mean/std using stable combination.
    """
    if simple:
        return df[f"{col}_mean"].mean(), df[f"{col}_std"].mean()

    # pooled mean/std based on combining mean/var/length
    def _combine_two(x1, x2):
        mean1, var1, n1 = x1
        mean2, var2, n2 = x2
        mean = mean1 * n1 / (n1 + n2) + mean2 * n2 / (n1 + n2)
        var = (var1 * n1 / (n1 + n2)
               + var2 * n2 / (n1 + n2)
               + (n1 * n2) / ((n1 + n2) ** 2) * np.power(mean1 - mean2, 2))
        return mean, var, (n1 + n2)

    means = list(df[f"{col}_mean"])
    vars_ = np.power(list(df[f"{col}_std"]), 2)
    lengths = list(df[f"{col}_length"])

    result = (means[0], vars_[0], lengths[0])
    for i in range(1, len(means)):
        result = _combine_two(result, (means[i], vars_[i], lengths[i]))
    pooled_mean, pooled_var, _ = result
    return pooled_mean, np.sqrt(pooled_var)


# ---------------------------------------------------------------------
# Resampling / PTB-XL preparation
# ---------------------------------------------------------------------

def resample_data(sigbufs: np.ndarray,
                  channel_labels: List[str],
                  fs: float,
                  target_fs: float,
                  channels: int = 8,
                  channel_stoi: Union[None, Dict[str, int]] = None,
                  skimage_transform: bool = True,
                  interpolation_order: int = 3) -> np.ndarray:
    """
    Resample multi-channel ECG to target_fs and optionally reorder/select channels via channel_stoi.
    Keeps original interpolation choices (skimage.transform.resize vs scipy.ndimage.zoom).
    """
    channel_labels = [c.lower() for c in channel_labels]
    factor = float(target_fs) / float(fs)
    timesteps_new = int(len(sigbufs) * factor)

    if channel_stoi is not None:
        data = np.zeros((timesteps_new, channels), dtype=np.float32)
        for i, cl in enumerate(channel_labels):
            if cl in channel_stoi and channel_stoi[cl] < channels:
                if skimage_transform:
                    data[:, channel_stoi[cl]] = transform.resize(
                        sigbufs[:, i], (timesteps_new,), order=interpolation_order
                    ).astype(np.float32)
                else:
                    data[:, channel_stoi[cl]] = zoom(
                        sigbufs[:, i], timesteps_new / len(sigbufs), order=interpolation_order
                    ).astype(np.float32)
    else:
        if skimage_transform:
            data = transform.resize(sigbufs, (timesteps_new, channels), order=interpolation_order).astype(np.float32)
        else:
            data = zoom(sigbufs, (timesteps_new / len(sigbufs), 1), order=interpolation_order).astype(np.float32)

    return data


def filter_ptb_xl(df: pd.DataFrame,
                  min_cnt: int = 0,
                  categories: List[str] = None) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Filter PTB-XL label columns so that only labels with count >= min_cnt remain.
    Creates '<category>_filtered' and '<category>_filtered_numeric' columns for each category.
    Returns (filtered_df, label_vocab_per_category).
    """
    if categories is None:
        categories = [
            "label_all", "label_diag", "label_form", "label_rhythm",
            "label_diag_subclass", "label_diag_superclass"
        ]

    def _select_labels(labels: Iterable[List[str]], min_cnt_: int = 0) -> List[str]:
        lbl, cnt = np.unique([item for sublist in list(labels) for item in sublist], return_counts=True)
        return list(lbl[np.where(cnt >= min_cnt_)[0]])

    df_ptb_xl = df.copy()
    lbl_itos_ptb_xl: Dict[str, np.ndarray] = {}

    for selection in categories:
        label_selected = _select_labels(df_ptb_xl[selection], min_cnt_=min_cnt)
        filt_col = f"{selection}_filtered"
        df_ptb_xl[filt_col] = df_ptb_xl[selection].apply(lambda x: [y for y in x if y in label_selected])

        lbl_itos_ptb_xl[selection] = np.array(list(set([x for sublist in df_ptb_xl[filt_col] for x in sublist])))
        lbl_stoi = {s: i for i, s in enumerate(lbl_itos_ptb_xl[selection])}

        df_ptb_xl[f"{selection}_filtered_numeric"] = df_ptb_xl[filt_col].apply(lambda x: [lbl_stoi[y] for y in x])

    return df_ptb_xl, lbl_itos_ptb_xl


def prepare_data_ptb_xl(data_path: Path,
                        min_cnt: int = 0,
                        target_fs: int = 100,
                        channels: int = 8,
                        channel_stoi: Dict[str, int] = channel_stoi_default,
                        target_folder: Union[None, Path] = None,
                        skimage_transform: bool = True,
                        recreate_data: bool = True
                        ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Prepare PTB-XL dataset:
    - Read CSVs, build label columns and class mappings.
    - Resample signals to target_fs, select/reorder channels.
    - Save per-record .npy (if recreate_data) and compute dataset stats.
    - Return (df, label_vocab_dict, mean, std).
    """
    target_root_ptb_xl = Path(".") if target_folder is None else Path(target_folder)
    target_root_ptb_xl.mkdir(parents=True, exist_ok=True)

    if recreate_data:
        # read index
        ptb_xl_csv = Path(data_path) / "ptbxl_database.csv"
        df_ptb_xl = pd.read_csv(ptb_xl_csv, index_col="ecg_id")
        # restore dict from string safely (keeps original behavior for 'nan' tokens)
        df_ptb_xl.scp_codes = df_ptb_xl.scp_codes.apply(lambda x: eval(str(x).replace("nan", "np.nan")))

        # label definitions
        ptb_xl_label_df = pd.read_csv(Path(data_path) / "scp_statements.csv").set_index("Unnamed: 0")
        ptb_xl_label_diag = ptb_xl_label_df[ptb_xl_label_df.diagnostic > 0]
        ptb_xl_label_form = ptb_xl_label_df[ptb_xl_label_df.form > 0]
        ptb_xl_label_rhythm = ptb_xl_label_df[ptb_xl_label_df.rhythm > 0]

        diag_class_mapping = {}
        diag_subclass_mapping = {}
        for sid, row in ptb_xl_label_diag.iterrows():
            if isinstance(row["diagnostic_class"], str):
                diag_class_mapping[sid] = row["diagnostic_class"]
            if isinstance(row["diagnostic_subclass"], str):
                diag_subclass_mapping[sid] = row["diagnostic_subclass"]

        # assemble label columns
        df_ptb_xl["label_all"] = df_ptb_xl.scp_codes.apply(lambda x: [y for y in x.keys()])
        df_ptb_xl["label_diag"] = df_ptb_xl.scp_codes.apply(
            lambda x: [y for y in x.keys() if y in ptb_xl_label_diag.index])
        df_ptb_xl["label_form"] = df_ptb_xl.scp_codes.apply(
            lambda x: [y for y in x.keys() if y in ptb_xl_label_form.index])
        df_ptb_xl["label_rhythm"] = df_ptb_xl.scp_codes.apply(
            lambda x: [y for y in x.keys() if y in ptb_xl_label_rhythm.index])

        df_ptb_xl["label_diag_subclass"] = df_ptb_xl.label_diag.apply(
            lambda x: [diag_subclass_mapping[y] for y in x if y in diag_subclass_mapping]
        )
        df_ptb_xl["label_diag_superclass"] = df_ptb_xl.label_diag.apply(
            lambda x: [diag_class_mapping[y] for y in x if y in diag_class_mapping]
        )

        df_ptb_xl["dataset"] = "ptb_xl"

        # filter labels by frequency threshold (can be re-applied later)
        df_ptb_xl, lbl_itos_ptb_xl = filter_ptb_xl(df_ptb_xl, min_cnt=min_cnt)

        # create per-record .npy with resampling/channel-selection
        filenames: List[Path] = []
        for _, row in tqdm(list(df_ptb_xl.iterrows())):
            filename_path = Path(data_path / row["filename_lr"]) if target_fs <= 100 else Path(
                data_path / row["filename_hr"])
            out_name = Path(filename_path.stem + ".npy")
            target_file = target_root_ptb_xl / out_name

            filenames.append(out_name)
            if target_file.exists():
                continue

            sigbufs, header = wfdb.rdsamp(str(filename_path))
            data = resample_data(
                sigbufs=sigbufs,
                channel_stoi=channel_stoi,
                channel_labels=header["sig_name"],
                fs=header["fs"],
                target_fs=target_fs,
                channels=channels,
                skimage_transform=skimage_transform
            )
            assert target_fs <= header["fs"], "target_fs must be <= original sampling rate"
            np.save(target_file, data)

        df_ptb_xl["data"] = filenames

        # per-file stats
        dataset_add_mean_col(df_ptb_xl, data_folder=target_root_ptb_xl)
        dataset_add_std_col(df_ptb_xl, data_folder=target_root_ptb_xl)
        dataset_add_length_col(df_ptb_xl, data_folder=target_root_ptb_xl)

        mean_ptb_xl, std_ptb_xl = dataset_get_stats(df_ptb_xl)

        # persist artifacts
        save_dataset(df_ptb_xl, lbl_itos_ptb_xl, mean_ptb_xl, std_ptb_xl, target_root_ptb_xl)

    else:
        # load previously saved artifacts
        df_ptb_xl, lbl_itos_ptb_xl, mean_ptb_xl, std_ptb_xl = load_dataset(target_root_ptb_xl, df_mapped=False)

    return df_ptb_xl, lbl_itos_ptb_xl, mean_ptb_xl, std_ptb_xl


def load_raw_data(df: pd.DataFrame, sampling_rate: int, path: str) -> np.ndarray:
    """
    Load raw WFDB signals referenced by df at a given sampling_rate (100 or 500),
    returning an array of shape [N, T, C].
    """
    if sampling_rate == 100:
        pairs = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        pairs = [wfdb.rdsamp(path + f) for f in df.filename_hr]
    data = np.array([signal for signal, _ in pairs])
    return data


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------

if __name__ == '__main__':
    # You can toggle target_fs = 100 or 500 as needed
    # target_fs = 100
    target_fs = 500

    data_root = Path('../Dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/')
    target_root = Path("./dataset")
    target_folder = target_root / f"PTBXL_fs{target_fs}"
    target_folder.mkdir(parents=True, exist_ok=True)

    df, lbl_itos, mean, std = prepare_data_ptb_xl(
        data_path=data_root,
        target_fs=target_fs,
        channels=12,
        channel_stoi=channel_stoi_default,
        target_folder=target_folder
    )

    # Reformat everything as memmap for efficiency
    reformat_as_memmap(
        df,
        target_folder / "memmap.npy",
        data_folder=target_folder,
        delete_npys=True
    )
