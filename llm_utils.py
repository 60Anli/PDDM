import os
from pathlib import Path

import numpy as np
import torch


DEFAULT_LLM_CONFIG = {
    "enabled": False,
    "mode": "cache",
    "cache_dir": "./data/llm_embeddings",
    "embedding_dim": 768,
    "proj_dim": 32,
    "num_mask_variants": 4,
    "missing_ratio_min": 0.05,
    "missing_ratio_max": 0.5,
    "require_cache": True,
    "max_prompt_values": 48,
    "cache_dtype": "float32",
}


def resolve_llm_config(config):
    cfg = dict(DEFAULT_LLM_CONFIG)
    if config is not None:
        cfg.update(config)
    cfg["enabled"] = bool(cfg.get("enabled", False))
    cfg["mode"] = str(cfg.get("mode", "cache")).lower()
    cfg["embedding_dim"] = int(cfg.get("embedding_dim", 768))
    cfg["proj_dim"] = int(cfg.get("proj_dim", 32))
    cfg["num_mask_variants"] = max(1, int(cfg.get("num_mask_variants", 1)))
    cfg["missing_ratio_min"] = float(cfg.get("missing_ratio_min", 0.05))
    cfg["missing_ratio_max"] = float(cfg.get("missing_ratio_max", 0.5))
    cfg["require_cache"] = bool(cfg.get("require_cache", True))
    cfg["max_prompt_values"] = max(1, int(cfg.get("max_prompt_values", 48)))
    cfg["cache_dtype"] = str(cfg.get("cache_dtype", "float32")).lower()
    if cfg["cache_dtype"] not in {"float16", "float32"}:
        cfg["cache_dtype"] = "float32"
    return cfg


def physio_split_tag(split_name, seed, missing_ratio, nfold):
    return f"seed{seed}_miss{missing_ratio}_fold{nfold}/{split_name}"


def pm25_split_tag(split_name, validindex):
    return f"valid{validindex}/{split_name}"


def sequence_imputation_split_tag(
    split_name,
    eval_length,
    target_strategy,
    missing_pattern,
    missing_ratio,
    dataset_name=None,
):
    ratio_text = str(missing_ratio).replace(".", "p")
    suffix = ""
    if dataset_name:
        safe_name = "".join(
            ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(dataset_name)
        )
        suffix = f"_data{safe_name}"
    return (
        f"el{int(eval_length)}_tgt{str(target_strategy).lower()}_pat{str(missing_pattern).lower()}"
        f"_miss{ratio_text}{suffix}/{split_name}"
    )


def forecasting_split_tag(split_name, seq_len, pred_len, features, data_name=None):
    suffix = ""
    if data_name:
        safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(data_name))
        suffix = f"_data{safe_name}"
    return f"sl{int(seq_len)}_pl{int(pred_len)}_feat{features}{suffix}/{split_name}"


def llm_cache_path(llm_config, dataset_name, split_tag, sample_id, variant_id):
    return (
        Path(llm_config["cache_dir"])
        / dataset_name
        / split_tag
        / f"v{int(variant_id)}"
        / f"{int(sample_id)}.pt"
    )


def forecasting_cache_file(llm_config, dataset_name, split_tag):
    return Path(llm_config["cache_dir"]) / dataset_name / split_tag / "embeddings.npy"


def choose_variant(llm_config, is_train):
    if not is_train or int(llm_config.get("num_mask_variants", 1)) <= 1:
        return 0
    return int(np.random.randint(0, llm_config["num_mask_variants"]))


def deterministic_rng(base_seed, sample_id, variant_id):
    seed = int(base_seed) + int(sample_id) * 1009 + int(variant_id) * 9176
    seed = seed % (2**32 - 1)
    return np.random.RandomState(seed)


def _ensure_has_target(cond_mask, observed_mask, rng):
    observed_idx = np.flatnonzero(observed_mask.reshape(-1) > 0.5)
    if len(observed_idx) <= 1:
        return cond_mask

    target_mask = (observed_mask > 0.5) & (cond_mask < 0.5)
    if target_mask.any():
        return cond_mask

    cond_flat = cond_mask.reshape(-1)
    hide_idx = rng.choice(observed_idx, 1, replace=False)
    cond_flat[hide_idx] = 0.0
    return cond_flat.reshape(cond_mask.shape)


def make_block_cond_mask(observed_mask, rng, seed_prob=0.15, min_seq=12, max_seq=24, noise_prob=0.05):
    observed_mask = np.asarray(observed_mask, dtype=np.float32)
    cond_mask = observed_mask.copy()
    if observed_mask.ndim != 2:
        return cond_mask

    time_dim, feature_dim = observed_mask.shape
    mask_seed = rng.rand(time_dim, feature_dim) < float(seed_prob)

    for col in range(feature_dim):
        seed_idx = np.flatnonzero(mask_seed[:, col])
        if len(seed_idx) == 0:
            continue

        if max_seq > min_seq:
            fault_len = min_seq + rng.randint(max_seq - min_seq + 1, size=len(seed_idx))
        else:
            fault_len = np.full(len(seed_idx), min_seq, dtype=np.int64)

        masked_index = []
        for start, span in zip(seed_idx, fault_len):
            masked_index.append(np.arange(start, start + int(span)))
        masked_index = np.unique(np.concatenate(masked_index))
        masked_index = np.clip(masked_index, 0, time_dim - 1)
        cond_mask[masked_index, col] = 0.0

    if noise_prob > 0:
        noise_mask = (rng.rand(time_dim, feature_dim) < float(noise_prob)).astype(np.float32)
        cond_mask = cond_mask * (1.0 - noise_mask)

    cond_mask = np.minimum(cond_mask, observed_mask)
    return _ensure_has_target(cond_mask.astype(np.float32), observed_mask, rng)


def make_cond_mask(
    observed_mask,
    gt_mask=None,
    hist_mask=None,
    split_name="train",
    target_strategy="random",
    sample_id=0,
    variant_id=0,
    base_seed=0,
    llm_config=None,
):
    observed_mask = np.asarray(observed_mask, dtype=np.float32)
    strategy = str(target_strategy).lower()

    if gt_mask is not None and strategy in {"test", "forecast", "forecasting"}:
        return np.asarray(gt_mask, dtype=np.float32).copy()

    if split_name != "train":
        if gt_mask is None:
            return observed_mask.copy()
        return np.asarray(gt_mask, dtype=np.float32).copy()

    cfg = resolve_llm_config(llm_config)
    rng = deterministic_rng(base_seed, sample_id, variant_id)

    if strategy == "block":
        return make_block_cond_mask(observed_mask, rng)

    if strategy in ("historical", "mix") and hist_mask is not None:
        use_hist = strategy == "historical" or rng.rand() > 0.5
        if use_hist:
            cond_mask = observed_mask * np.asarray(hist_mask, dtype=np.float32)
            return _ensure_has_target(cond_mask.astype(np.float32), observed_mask, rng)

    obs_idx = np.flatnonzero(observed_mask.reshape(-1) > 0.5)
    cond_flat = observed_mask.reshape(-1).copy()
    if len(obs_idx) == 0:
        return cond_flat.reshape(observed_mask.shape).astype(np.float32)

    min_ratio = max(0.0, min(1.0, cfg["missing_ratio_min"]))
    max_ratio = max(min_ratio, min(1.0, cfg["missing_ratio_max"]))
    sample_ratio = rng.uniform(min_ratio, max_ratio)
    num_masked = int(round(len(obs_idx) * sample_ratio))
    num_masked = min(max(1, num_masked), max(1, len(obs_idx) - 1))
    hide_idx = rng.choice(obs_idx, num_masked, replace=False)
    cond_flat[hide_idx] = 0.0
    return cond_flat.reshape(observed_mask.shape).astype(np.float32)


def _format_float(value):
    if not np.isfinite(value):
        return "nan"
    return f"{float(value):.4g}"


def build_feature_prompt(
    dataset_name,
    feature_name,
    values,
    observed_mask,
    cond_mask,
    timepoints,
    max_prompt_values=48,
    task_name="imputation",
):
    values = np.asarray(values, dtype=np.float32)
    observed_mask = np.asarray(observed_mask, dtype=np.float32)
    cond_mask = np.asarray(cond_mask, dtype=np.float32)
    timepoints = np.asarray(timepoints)
    task = str(task_name).lower()

    known_idx = np.where((observed_mask > 0.5) & (cond_mask > 0.5))[0]
    target_idx = np.where((observed_mask > 0.5) & (cond_mask < 0.5))[0]
    known_values = values[known_idx]

    if len(known_values) > 0:
        mean = _format_float(np.mean(known_values))
        std = _format_float(np.std(known_values))
        min_v = _format_float(np.min(known_values))
        max_v = _format_float(np.max(known_values))
        first_v = _format_float(known_values[0])
        last_v = _format_float(known_values[-1])
        trend = _format_float(known_values[-1] - known_values[0])
    else:
        mean = std = min_v = max_v = first_v = last_v = trend = "nan"

    pairs = []
    for idx in known_idx[:max_prompt_values]:
        pairs.append(f"t{int(timepoints[idx])}={_format_float(values[idx])}")
    known_text = ", ".join(pairs) if pairs else "none"
    if len(known_idx) > max_prompt_values:
        known_text += f", ... ({len(known_idx) - max_prompt_values} more)"

    targets = [f"t{int(timepoints[idx])}" for idx in target_idx[:max_prompt_values]]
    target_text = ", ".join(targets) if targets else "none"
    if len(target_idx) > max_prompt_values:
        target_text += f", ... ({len(target_idx) - max_prompt_values} more)"

    if task == "forecasting":
        task_description = "represent the observed history of this time series for forecasting future values"
        target_label = "Future target positions"
    else:
        task_description = "represent this partially observed time series for imputation"
        target_label = "Missing target positions"

    return (
        f"Dataset: {dataset_name}. Variable: {feature_name}. "
        f"Task: {task_description}. "
        f"Known values: {known_text}. "
        f"{target_label}: {target_text}. "
        f"Known summary: count={len(known_idx)}, mean={mean}, std={std}, "
        f"min={min_v}, max={max_v}, first={first_v}, last={last_v}, trend={trend}."
    )


def build_prompts_for_sample(
    dataset_name,
    feature_names,
    observed_data,
    observed_mask,
    cond_mask,
    timepoints,
    max_prompt_values=48,
    task_name="imputation",
):
    prompts = []
    for k, feature_name in enumerate(feature_names):
        prompts.append(
            build_feature_prompt(
                dataset_name=dataset_name,
                feature_name=feature_name,
                values=observed_data[:, k],
                observed_mask=observed_mask[:, k],
                cond_mask=cond_mask[:, k],
                timepoints=timepoints,
                max_prompt_values=max_prompt_values,
                task_name=task_name,
            )
        )
    return prompts


def load_llm_cache(path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    if isinstance(payload, dict):
        embedding = payload["embedding"]
        cond_mask = payload["cond_mask"]
    else:
        embedding, cond_mask = payload

    if isinstance(embedding, torch.Tensor):
        embedding = embedding.detach().cpu().numpy()
    if isinstance(cond_mask, torch.Tensor):
        cond_mask = cond_mask.detach().cpu().numpy()
    return cond_mask.astype(np.float32), embedding.astype(np.float32)


def save_llm_cache(path, cond_mask, embedding):
    os.makedirs(Path(path).parent, exist_ok=True)
    payload = {
        "cond_mask": torch.as_tensor(cond_mask, dtype=torch.float32),
        "embedding": torch.as_tensor(embedding, dtype=torch.float32),
    }
    torch.save(payload, path)


def attach_llm_fields(
    sample,
    llm_config,
    dataset_name,
    split_tag,
    split_name,
    sample_id,
    feature_names,
    target_strategy="random",
    base_seed=0,
):
    cfg = resolve_llm_config(llm_config)
    if not cfg["enabled"] or cfg["mode"] != "cache":
        return sample

    is_train = split_name == "train"
    variant_id = choose_variant(cfg, is_train=is_train)
    observed_data = np.asarray(sample["observed_data"], dtype=np.float32)
    observed_mask = np.asarray(sample["observed_mask"], dtype=np.float32)
    gt_mask = np.asarray(sample.get("gt_mask", observed_mask), dtype=np.float32)
    hist_mask = sample.get("hist_mask", None)

    cond_mask = make_cond_mask(
        observed_mask=observed_mask,
        gt_mask=gt_mask,
        hist_mask=hist_mask,
        split_name=split_name,
        target_strategy=target_strategy,
        sample_id=sample_id,
        variant_id=variant_id,
        base_seed=base_seed,
        llm_config=cfg,
    )

    path = llm_cache_path(cfg, dataset_name, split_tag, sample_id, variant_id)
    if path.exists():
        cond_mask, embedding = load_llm_cache(path)
    elif cfg["require_cache"]:
        raise FileNotFoundError(
            f"LLM cache not found: {path}. Run generate_llm_embeddings.py first, "
            "or set model.llm.require_cache=false for a zero-embedding smoke run."
        )
    else:
        embedding = np.zeros((len(feature_names), cfg["embedding_dim"]), dtype=np.float32)

    sample["cond_mask"] = cond_mask.astype(np.float32)
    sample["llm_embedding"] = embedding.astype(np.float32)
    sample["llm_variant"] = np.array(variant_id, dtype=np.int64)
    return sample