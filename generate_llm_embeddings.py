import argparse
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from numpy.lib.format import open_memmap
from tqdm import tqdm

from llm_utils import (
    build_prompts_for_sample,
    forecasting_cache_file,
    llm_cache_path,
    make_cond_mask,
    resolve_llm_config,
    save_llm_cache,
)


FORECASTING_DATASETS = {"electricity", "traffic"}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate cached LLM embeddings for PDDM.")
    parser.add_argument("--dataset", choices=["physio", "pm25", "electricity", "traffic"], required=True)
    parser.add_argument("--config", default="base.yaml")
    parser.add_argument("--split", choices=["train", "valid", "val", "test", "all"], default="all")
    parser.add_argument("--backend", choices=["gpt2", "zeros"], default="gpt2")
    parser.add_argument("--model_name", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_features", type=int, default=36)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--nfold", type=int, default=0)
    parser.add_argument("--testmissingratio", type=float, default=0.1)
    parser.add_argument("--validationindex", type=int, default=0)
    parser.add_argument("--targetstrategy", default=None)

    parser.add_argument("--root_path", default="")
    parser.add_argument("--data_path", default="")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=0)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--features", choices=["M", "MS", "S"], default="M")
    parser.add_argument("--target", default="OT")
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--timeenc", type=int, default=0)
    parser.add_argument("--freq", default="h")
    return parser.parse_args()


def load_config(config_name):
    path = os.path.join("config", config_name)
    with open(path, "r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


def load_imputation_datasets(args, config):
    config_strategy = config.get("model", {}).get("target_strategy", "random")
    target_strategy = args.targetstrategy or config_strategy
    if args.dataset == "physio":
        from dataset_physio import get_datasets, attributes

        datasets = get_datasets(
            seed=args.seed,
            nfold=args.nfold,
            missing_ratio=args.testmissingratio,
            llm_config={"enabled": False},
            target_strategy=target_strategy,
        )
        feature_names = attributes
        return {"train": datasets[0], "valid": datasets[1], "test": datasets[2]}, feature_names

    from dataset_pm25 import get_datasets

    target_strategy = args.targetstrategy or "mix"
    datasets = get_datasets(
        validindex=args.validationindex,
        llm_config={"enabled": False},
        target_strategy=target_strategy,
    )
    feature_names = datasets[0].feature_names
    return {"train": datasets[0], "valid": datasets[1], "test": datasets[2]}, feature_names


def load_forecasting_datasets(args, config):
    from dataset_forecasting import Dataset_Custom

    root_path = args.root_path or os.path.join(".", "data", args.dataset)
    data_path = args.data_path or f"{args.dataset}.csv"
    model_cfg = config.get("model", {})
    target_strategy = args.targetstrategy or model_cfg.get("target_strategy", "test")

    common = dict(
        root_path=root_path,
        data_path=data_path,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        features=args.features,
        target=args.target,
        scale=bool(args.scale),
        timeenc=args.timeenc,
        freq=args.freq,
        llm_config={"enabled": False},
        dataset_name=args.dataset,
        target_strategy=target_strategy,
    )
    train_dataset = Dataset_Custom(flag="train", **common)
    val_dataset = Dataset_Custom(flag="val", scaler=train_dataset.scaler, **common)
    test_dataset = Dataset_Custom(flag="test", scaler=train_dataset.scaler, **common)
    return {"train": train_dataset, "val": val_dataset, "test": test_dataset}, train_dataset.feature_names


def load_datasets(args, config):
    if args.dataset in FORECASTING_DATASETS:
        return load_forecasting_datasets(args, config)
    return load_imputation_datasets(args, config)


def prepare_gpt2(args, llm_config):
    if args.backend == "zeros":
        return None, None, torch.device("cpu")

    from transformers import GPT2Model, GPT2Tokenizer

    model_name = args.model_name or llm_config.get("model_name", "gpt2")
    local_files_only = bool(llm_config.get("local_files_only", False))
    strict = bool(llm_config.get("strict_load", False))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = GPT2Model.from_pretrained(model_name, local_files_only=local_files_only).to(device)
    except Exception as exc:
        if strict:
            raise RuntimeError(f"Failed to load GPT-2 from {model_name}: {exc}") from exc
        raise
    model.eval()
    return tokenizer, model, device


def encode_prompts(prompts, tokenizer, model, device, batch_features, embedding_dim):
    if tokenizer is None or model is None:
        return np.zeros((len(prompts), embedding_dim), dtype=np.float32)

    outputs = []
    for start in range(0, len(prompts), batch_features):
        batch_prompts = prompts[start : start + batch_features]
        tokens = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            hidden = model(**tokens).last_hidden_state
            last_idx = tokens["attention_mask"].sum(dim=1) - 1
            batch_emb = hidden[torch.arange(hidden.shape[0], device=device), last_idx]
        outputs.append(batch_emb.detach().cpu())
    return torch.cat(outputs, dim=0).numpy().astype(np.float32)


def sample_arrays(sample):
    return (
        np.asarray(sample["observed_data"], dtype=np.float32),
        np.asarray(sample["observed_mask"], dtype=np.float32),
        np.asarray(sample["gt_mask"], dtype=np.float32),
        np.asarray(sample["timepoints"]),
        sample.get("hist_mask", None),
    )


def generate_imputation_split(args, dataset_name, dataset, feature_names, llm_config, tokenizer, model, device):
    split_name = dataset.split_name if hasattr(dataset, "split_name") else dataset.mode
    split_tag = dataset.split_tag
    target_strategy = getattr(dataset, "target_strategy", "random")
    base_seed = getattr(dataset, "seed", getattr(dataset, "validindex", 0))

    variants = range(llm_config["num_mask_variants"]) if split_name == "train" else range(1)
    total = len(dataset) * len(list(variants))
    variants = range(llm_config["num_mask_variants"]) if split_name == "train" else range(1)

    with tqdm(total=total, desc=f"{dataset_name}:{split_name}") as pbar:
        for sample_id in range(len(dataset)):
            sample = dataset[sample_id]
            observed_data, observed_mask, gt_mask, timepoints, hist_mask = sample_arrays(sample)
            for variant_id in variants:
                path = llm_cache_path(llm_config, dataset_name, split_tag, sample_id, variant_id)
                if path.exists() and not args.overwrite:
                    pbar.update(1)
                    continue

                cond_mask = make_cond_mask(
                    observed_mask=observed_mask,
                    gt_mask=gt_mask,
                    hist_mask=hist_mask,
                    split_name=split_name,
                    target_strategy=target_strategy,
                    sample_id=sample_id,
                    variant_id=variant_id,
                    base_seed=base_seed,
                    llm_config=llm_config,
                )
                prompts = build_prompts_for_sample(
                    dataset_name=dataset_name,
                    feature_names=feature_names,
                    observed_data=observed_data,
                    observed_mask=observed_mask,
                    cond_mask=cond_mask,
                    timepoints=timepoints,
                    max_prompt_values=llm_config["max_prompt_values"],
                    task_name="imputation",
                )
                embedding = encode_prompts(
                    prompts=prompts,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    batch_features=args.batch_features,
                    embedding_dim=llm_config["embedding_dim"],
                )
                save_llm_cache(path, cond_mask, embedding)
                pbar.update(1)


def generate_forecasting_split(args, dataset_name, dataset, feature_names, llm_config, tokenizer, model, device):
    split_name = dataset.flag
    split_tag = dataset.split_tag
    target_strategy = getattr(dataset, "target_strategy", "test")
    cache_file = forecasting_cache_file(llm_config, dataset_name, split_tag)
    expected_shape = (len(dataset), len(feature_names), llm_config["embedding_dim"])

    if cache_file.exists() and not args.overwrite:
        existing = np.load(cache_file, mmap_mode="r")
        if tuple(existing.shape) == expected_shape:
            print(f"Skip existing forecasting cache: {cache_file}")
            return
        print(f"Rebuilding forecasting cache with new shape: {cache_file}")

    os.makedirs(cache_file.parent, exist_ok=True)
    storage_dtype = np.float16 if llm_config.get("cache_dtype", "float32") == "float16" else np.float32
    cache_array = open_memmap(cache_file, mode="w+", dtype=storage_dtype, shape=expected_shape)

    with tqdm(total=len(dataset), desc=f"{dataset_name}:{split_name}") as pbar:
        for sample_id in range(len(dataset)):
            sample = dataset[sample_id]
            observed_data, observed_mask, gt_mask, timepoints, _ = sample_arrays(sample)
            cond_mask = make_cond_mask(
                observed_mask=observed_mask,
                gt_mask=gt_mask,
                split_name=split_name,
                target_strategy=target_strategy,
                sample_id=sample_id,
                variant_id=0,
                base_seed=0,
                llm_config=llm_config,
            )
            prompts = build_prompts_for_sample(
                dataset_name=dataset_name,
                feature_names=feature_names,
                observed_data=observed_data,
                observed_mask=observed_mask,
                cond_mask=cond_mask,
                timepoints=timepoints,
                max_prompt_values=llm_config["max_prompt_values"],
                task_name="forecasting",
            )
            embedding = encode_prompts(
                prompts=prompts,
                tokenizer=tokenizer,
                model=model,
                device=device,
                batch_features=args.batch_features,
                embedding_dim=llm_config["embedding_dim"],
            )
            cache_array[sample_id] = embedding.astype(storage_dtype, copy=False)
            pbar.update(1)
    cache_array.flush()


def main():
    args = parse_args()
    config = load_config(args.config)
    llm_config = resolve_llm_config(config.get("model", {}).get("llm", {}))
    llm_config["enabled"] = True
    llm_config["mode"] = "cache"

    datasets, feature_names = load_datasets(args, config)
    tokenizer, model, device = prepare_gpt2(args, llm_config)

    if args.split == "all":
        splits = list(datasets.keys())
    else:
        split = args.split
        if args.dataset in FORECASTING_DATASETS and split == "valid":
            split = "val"
        if split not in datasets:
            raise KeyError(f"Split '{split}' is not available for dataset {args.dataset}. Available: {list(datasets.keys())}")
        splits = [split]

    for split in splits:
        if args.dataset in FORECASTING_DATASETS:
            generate_forecasting_split(
                args=args,
                dataset_name=args.dataset,
                dataset=datasets[split],
                feature_names=feature_names,
                llm_config=llm_config,
                tokenizer=tokenizer,
                model=model,
                device=device,
            )
        else:
            generate_imputation_split(
                args=args,
                dataset_name=args.dataset,
                dataset=datasets[split],
                feature_names=feature_names,
                llm_config=llm_config,
                tokenizer=tokenizer,
                model=model,
                device=device,
            )


if __name__ == "__main__":
    main()