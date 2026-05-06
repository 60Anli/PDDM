import os
import pickle as pk

import numpy as np
from torch.utils.data import DataLoader, Dataset

from llm_utils import attach_llm_fields, resolve_llm_config, sequence_imputation_split_tag


def generate_eval_mask(shape, p=0.0015, p_noise=0.05, min_seq=12, max_seq=24):
    rand = np.random.rand
    randint = np.random.randint
    mask = rand(*shape) < p
    time_dim, feature_dim = shape

    for col in range(feature_dim):
        idxs = np.flatnonzero(mask[:, col])
        if len(idxs) == 0:
            continue
        if max_seq > min_seq:
            fault_len = min_seq + randint(max_seq - min_seq + 1, size=len(idxs))
        else:
            fault_len = np.full(len(idxs), min_seq, dtype=np.int64)
        expanded = []
        for idx, span in zip(idxs, fault_len):
            expanded.append(np.arange(idx, idx + int(span)))
        expanded = np.unique(np.concatenate(expanded))
        expanded = np.clip(expanded, 0, time_dim - 1)
        mask[expanded, col] = True

    if p_noise > 0:
        mask = mask | (rand(*shape) < p_noise)
    return mask.astype(np.float32)


class WeatherImputationDataset(Dataset):
    def __init__(
        self,
        data,
        eval_length=24,
        missing_ratio=0.0015,
        missing_pattern="block",
        is_train=True,
        llm_config=None,
        split_name="train",
        split_tag=None,
        target_strategy="random",
        feature_names=None,
        base_seed=0,
    ):
        self.data = np.asarray(data, dtype=np.float32)
        self.eval_length = int(eval_length)
        self.missing_ratio = float(missing_ratio)
        self.missing_pattern = str(missing_pattern).lower()
        self.is_train = bool(is_train)
        self.llm_config = resolve_llm_config(llm_config)
        self.split_name = split_name
        self.target_strategy = target_strategy
        self.base_seed = int(base_seed)
        self.feature_names = feature_names or [f"feature_{idx}" for idx in range(self.data.shape[1])]
        self.split_tag = split_tag or sequence_imputation_split_tag(
            split_name=split_name,
            eval_length=self.eval_length,
            target_strategy=target_strategy,
            missing_pattern=self.missing_pattern,
            missing_ratio=self.missing_ratio,
            dataset_name="weather",
        )

        self.observed_mask = (~np.isnan(self.data)).astype(np.float32)
        self.data = np.nan_to_num(self.data).astype(np.float32)

        if self.is_train:
            self.gt_mask_global = self.observed_mask.copy()
        else:
            if self.missing_pattern == "block":
                seed_prob = self.missing_ratio if self.missing_ratio > 0 else 0.0015
                eval_mask = generate_eval_mask(
                    self.data.shape,
                    p=seed_prob,
                    p_noise=0.05,
                    min_seq=12,
                    max_seq=48,
                )
            else:
                point_prob = self.missing_ratio if self.missing_ratio > 0 else 0.2
                eval_mask = generate_eval_mask(
                    self.data.shape,
                    p=0.0,
                    p_noise=point_prob,
                    min_seq=1,
                    max_seq=1,
                )
            self.gt_mask_global = self.observed_mask * (1.0 - eval_mask)

        self.total_length = self.data.shape[0]
        self.num_samples = max(0, self.total_length - self.eval_length + 1)
        if self.is_train:
            self.sample_indices = np.arange(self.num_samples)
        else:
            self.sample_indices = np.arange(0, self.num_samples, self.eval_length)
        self.num_samples = len(self.sample_indices)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start_idx = int(self.sample_indices[idx])
        end_idx = start_idx + self.eval_length
        sample = {
            "observed_data": self.data[start_idx:end_idx].astype(np.float32),
            "observed_mask": self.observed_mask[start_idx:end_idx].astype(np.float32),
            "gt_mask": self.gt_mask_global[start_idx:end_idx].astype(np.float32),
            "hist_mask": self.gt_mask_global[start_idx:end_idx].astype(np.float32),
            "timepoints": np.arange(self.eval_length, dtype=np.float32),
            "cut_length": np.array(0, dtype=np.int64),
        }
        sample = attach_llm_fields(
            sample,
            llm_config=self.llm_config,
            dataset_name="weather",
            split_tag=self.split_tag,
            split_name=self.split_name,
            sample_id=start_idx,
            feature_names=self.feature_names,
            target_strategy=self.target_strategy,
            base_seed=self.base_seed,
        )
        return sample


def get_weather_dataloader(
    data_path,
    batch_size,
    eval_length,
    missing_ratio,
    missing_pattern,
    num_workers=0,
    llm_config=None,
    target_strategy="random",
):
    train_data = pk.load(open(os.path.join(data_path, "train_set.pkl"), "rb"))
    val_data = pk.load(open(os.path.join(data_path, "val_set.pkl"), "rb"))
    test_data = pk.load(open(os.path.join(data_path, "test_set.pkl"), "rb"))

    scaler_path = os.path.join(data_path, "scaler_params.pkl")
    if not os.path.exists(scaler_path):
        scaler_path = os.path.join(data_path, "scaler.pkl")
    scaler_params = pk.load(open(scaler_path, "rb"))

    feature_names = [f"feature_{idx}" for idx in range(train_data.shape[1])]
    train_dataset = WeatherImputationDataset(
        train_data,
        eval_length=eval_length,
        missing_ratio=missing_ratio,
        missing_pattern=missing_pattern,
        is_train=True,
        llm_config=llm_config,
        split_name="train",
        target_strategy=target_strategy,
        feature_names=feature_names,
    )
    val_dataset = WeatherImputationDataset(
        val_data,
        eval_length=eval_length,
        missing_ratio=missing_ratio,
        missing_pattern=missing_pattern,
        is_train=False,
        llm_config=llm_config,
        split_name="valid",
        target_strategy=target_strategy,
        feature_names=feature_names,
    )
    test_dataset = WeatherImputationDataset(
        test_data,
        eval_length=eval_length,
        missing_ratio=missing_ratio,
        missing_pattern=missing_pattern,
        is_train=False,
        llm_config=llm_config,
        split_name="test",
        target_strategy=target_strategy,
        feature_names=feature_names,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, scaler_params
