import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from llm_utils import forecasting_cache_file, forecasting_split_tag, resolve_llm_config


@dataclass
class TrainOnlyStandardScaler:
    """Feature-wise standard scaler fitted on the chronological train split only."""

    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def fit(self, values: np.ndarray, mask: np.ndarray | None = None):
        if mask is None:
            mean = values.mean(axis=0)
            std = values.std(axis=0)
        else:
            valid_count = np.clip(mask.sum(axis=0), a_min=1.0, a_max=None)
            mean = (values * mask).sum(axis=0) / valid_count
            centered = (values - mean) * mask
            std = np.sqrt((centered ** 2).sum(axis=0) / valid_count)

        std = np.where(std < 1e-6, 1.0, std)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler must be fitted before calling transform().")
        return ((values - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler must be fitted before calling inverse_transform().")
        return (values * self.std + self.mean).astype(np.float32)


def _looks_like_datetime(series: pd.Series) -> bool:
    if np.issubdtype(series.dtype, np.datetime64):
        return True
    if not (pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype)):
        return False
    parsed = pd.to_datetime(series, errors="coerce")
    return bool(parsed.notna().all())


class Dataset_Custom(Dataset):
    """
    Benchmark-style long-term forecasting dataset.

    The full sequence is [history, future], where:
    - observed_mask marks which raw values exist in the CSV
    - gt_mask marks the conditioning region for forecasting:
      history is visible, future is hidden
    """

    def __init__(
        self,
        root_path,
        data_path,
        flag="train",
        seq_len=96,
        label_len=0,
        pred_len=96,
        features="M",
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
        scaler=None,
        split_ratio=(0.7, 0.1, 0.2),
        llm_config=None,
        dataset_name="forecasting",
        split_tag=None,
        target_strategy="test",
    ):
        if flag not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported flag: {flag}")

        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag
        self.mode = flag
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.features = features
        self.target = target
        self.scale = bool(scale)
        self.timeenc = timeenc
        self.freq = freq
        self.split_ratio = split_ratio
        self.llm_config = resolve_llm_config(llm_config)
        self.dataset_name = dataset_name
        self.target_strategy = target_strategy
        self.data_name = Path(self.data_path).stem
        self.split_tag = split_tag or forecasting_split_tag(
            split_name=flag,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            features=self.features,
            data_name=self.data_name,
        )

        self.scaler = scaler
        self.llm_embeddings = None
        self._read_data()

    def _read_data(self):
        csv_path = os.path.join(self.root_path, self.data_path)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")

        df_raw = pd.read_csv(csv_path)
        if df_raw.shape[1] < 2:
            raise ValueError(f"Expected at least 2 columns in {csv_path}, got {df_raw.shape[1]}")

        data_frame = df_raw.copy()
        first_column = data_frame.iloc[:, 0]
        if str(data_frame.columns[0]).lower() in {"date", "datetime", "time", "timestamp"} or _looks_like_datetime(first_column):
            data_frame = data_frame.iloc[:, 1:]

        if self.features == "M":
            feature_frame = data_frame
            self.eval_feature_mask = np.ones((1, data_frame.shape[1]), dtype=np.float32)
        elif self.features == "MS":
            if self.target not in data_frame.columns:
                raise ValueError(f"Target column '{self.target}' was not found in {self.data_path}")
            feature_frame = data_frame
            target_index = data_frame.columns.get_loc(self.target)
            self.eval_feature_mask = np.zeros((1, data_frame.shape[1]), dtype=np.float32)
            self.eval_feature_mask[0, target_index] = 1.0
        elif self.features == "S":
            if self.target not in data_frame.columns:
                raise ValueError(f"Target column '{self.target}' was not found in {self.data_path}")
            feature_frame = data_frame[[self.target]]
            self.eval_feature_mask = np.ones((1, 1), dtype=np.float32)
        else:
            raise ValueError(f"Unsupported feature mode: {self.features}")

        self.feature_names = [str(col) for col in feature_frame.columns]
        raw_values = feature_frame.to_numpy(dtype=np.float32)
        observed_mask = (~np.isnan(raw_values)).astype(np.float32)
        filled_values = np.nan_to_num(raw_values, nan=0.0).astype(np.float32)

        total_length = len(filled_values)
        num_train = int(total_length * self.split_ratio[0])
        num_test = int(total_length * self.split_ratio[2])
        num_val = total_length - num_train - num_test

        if num_train <= self.seq_len or num_val <= 0 or num_test <= 0:
            raise ValueError(
                "Dataset is too short for the requested 7:1:2 split and forecasting window."
            )

        border1s = [
            0,
            num_train - self.seq_len,
            total_length - num_test - self.seq_len,
        ]
        border2s = [
            num_train,
            num_train + num_val,
            total_length,
        ]
        split_index = {"train": 0, "val": 1, "test": 2}[self.flag]
        border1 = border1s[split_index]
        border2 = border2s[split_index]

        if self.scale:
            if self.scaler is None:
                self.scaler = TrainOnlyStandardScaler().fit(
                    filled_values[:num_train], observed_mask[:num_train]
                )
            scaled_values = self.scaler.transform(filled_values)
        else:
            scaled_values = filled_values

        scaled_values = (scaled_values * observed_mask).astype(np.float32)

        self.data_x = scaled_values[border1:border2]
        self.data_mask = observed_mask[border1:border2]
        self.total_seq_len = self.seq_len + self.pred_len
        self.feature_dim = self.data_x.shape[1]
        self.border1 = border1
        self.border2 = border2

        available = len(self.data_x) - self.total_seq_len + 1
        if available <= 0:
            raise ValueError(
                f"Split '{self.flag}' is too short for seq_len={self.seq_len} and pred_len={self.pred_len}."
            )
        self.available_windows = available
        self._load_llm_embeddings()

    def _load_llm_embeddings(self):
        if not self.llm_config["enabled"] or self.llm_config["mode"] != "cache":
            self.llm_embeddings = None
            return

        cache_file = forecasting_cache_file(self.llm_config, self.dataset_name, self.split_tag)
        if not cache_file.exists():
            if self.llm_config["require_cache"]:
                raise FileNotFoundError(
                    f"Forecasting LLM cache not found: {cache_file}. "
                    "Run generate_llm_embeddings.py for this dataset and split first."
                )
            self.llm_embeddings = None
            return

        llm_embeddings = np.load(cache_file, mmap_mode="r")
        expected_shape = (self.available_windows, self.feature_dim, self.llm_config["embedding_dim"])
        if tuple(llm_embeddings.shape) != expected_shape:
            raise ValueError(
                f"Forecasting LLM cache shape mismatch for {cache_file}. "
                f"Expected {expected_shape}, got {tuple(llm_embeddings.shape)}."
            )
        self.llm_embeddings = llm_embeddings

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        history = self.data_x[s_begin:s_end]
        future = self.data_x[r_begin:r_end]
        history_mask = self.data_mask[s_begin:s_end]
        future_mask = self.data_mask[r_begin:r_end]

        full_seq = np.concatenate([history, future], axis=0).astype(np.float32)
        observed_mask = np.concatenate([history_mask, future_mask], axis=0).astype(np.float32)
        gt_mask = np.concatenate(
            [history_mask, np.zeros_like(future_mask, dtype=np.float32)], axis=0
        ).astype(np.float32)

        sample = {
            "observed_data": full_seq,
            "observed_mask": observed_mask,
            "gt_mask": gt_mask,
            "hist_mask": gt_mask,
            "timepoints": np.arange(self.total_seq_len, dtype=np.float32),
            "cut_length": np.int64(0),
            "x_enc": history.astype(np.float32),
            "y_gt": future.astype(np.float32),
            "obs_mask": gt_mask.copy(),
            "eval_feature_mask": np.repeat(
                self.eval_feature_mask, self.total_seq_len, axis=0
            ).astype(np.float32),
        }
        if self.llm_embeddings is not None:
            sample["cond_mask"] = gt_mask.copy()
            sample["llm_embedding"] = np.asarray(self.llm_embeddings[index], dtype=np.float32)
        return sample

    def __len__(self):
        return self.available_windows


def data_provider(args, flag, scaler=None):
    dataset = Dataset_Custom(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        features=args.features,
        target=args.target,
        scale=args.scale,
        timeenc=args.timeenc,
        freq=args.freq,
        scaler=scaler,
        llm_config=getattr(args, "llm_config", None),
        dataset_name=getattr(args, "dataset", Path(args.data_path).stem),
        target_strategy=getattr(args, "target_strategy", "test"),
    )

    shuffle_flag = flag == "train"
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=False,
    )
    return dataset, data_loader


def get_forecasting_dataloader(args, device):
    train_dataset, train_loader = data_provider(args, flag="train")
    val_dataset, val_loader = data_provider(args, flag="val", scaler=train_dataset.scaler)
    test_dataset, test_loader = data_provider(args, flag="test", scaler=train_dataset.scaler)

    if train_dataset.scaler is None:
        mean_scaler = torch.zeros(train_dataset.feature_dim, device=device).float()
        scaler = torch.ones(train_dataset.feature_dim, device=device).float()
    else:
        mean_scaler = torch.from_numpy(train_dataset.scaler.mean).to(device).float()
        scaler = torch.from_numpy(train_dataset.scaler.std).to(device).float()

    return (
        train_loader,
        val_loader,
        test_loader,
        scaler,
        mean_scaler,
        train_dataset.feature_dim,
    )