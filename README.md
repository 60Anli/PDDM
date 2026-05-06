# PDDM Coarse Prior with GPT-2

This repository contains a PDDM-based time-series imputation branch that adds a GPT-2 coarse prior (`x_prior`) while keeping the diffusion model responsible for final refinement.

## Current Idea

Instead of injecting a full LLM feature branch into the diffusion backbone, this version only uses GPT-2 to produce a coarse imputation prior:

- observed values and `cond_mask` are encoded by a runtime GPT-2 conditioner
- the conditioner predicts a coarse prior `x_prior`
- `x_prior` is injected only on target positions through a zero-initialized residual gate
- the diffusion model still performs the final denoising and imputation
- training includes an auxiliary `prior_loss`

This design is intended to reduce the risk that LLM features directly disturb the original PDDM denoising path.

## Main Code Files

- `runtime_llm_conditioner.py`: GPT-2-based coarse prior generator
- `多时间步和层间main_model.py`: imputation model with `x_prior` and `prior_loss`
- `多时间步和层间diff_model对L升维.py`: diffusion backbone without the old `llm_info` branch
- `config/base_llm.yaml`: main experiment configuration

## Datasets Supported

Imputation:
- Physio
- PM2.5
- Weather
- ETTm1

Forecasting scripts are also kept in the repository, but the main focus of this branch is the coarse-prior imputation setting.

## Environment

Recommended environment:
- Python in conda env `torch230cuda121`
- PyTorch
- transformers
- safetensors
- pandas
- pyyaml
- tqdm

GPT-2 is loaded locally from:
- `./pretrained_models/gpt2`

## Run Commands

### Physio
```powershell
python exe_physio.py --config base_llm.yaml --device cuda:0 --seed 1 --nfold 0 --testmissingratio 0.1 --nsample 100
```

### PM2.5
```powershell
python exe_pm25.py --config base_llm.yaml --device cuda:0 --validationindex 0 --targetstrategy random --nsample 100
```

### Weather Random Missing
```powershell
python exe_weather.py --config base_llm.yaml --device cuda:0 --data_path ./data/Weather --eval_length 24 --missing_ratio 0.2 --missing_pattern random --target_strategy random --nsample 50
```

### Weather Block Missing
```powershell
python exe_weather.py --config base_llm.yaml --device cuda:0 --data_path ./data/Weather --eval_length 24 --missing_ratio 0.0015 --missing_pattern block --target_strategy block --nsample 50
```

### ETTm1 Random Missing
```powershell
python exe_ettm1.py --config base_llm.yaml --device cuda:0 --data_path ./data/ETT_processed/ETTm1 --raw_data_path ./data/ETT_raw/ETTm1.csv --eval_length 24 --missing_ratio 0.2 --missing_pattern random --target_strategy random --nsample 50
```

### ETTm1 Block Missing
```powershell
python exe_ettm1.py --config base_llm.yaml --device cuda:0 --data_path ./data/ETT_processed/ETTm1 --raw_data_path ./data/ETT_raw/ETTm1.csv --eval_length 24 --missing_ratio 0.0015 --missing_pattern block --target_strategy block --nsample 50
```

## Notes on Versioning

This repository intentionally does not track:
- dataset files
- experiment outputs under `save/`
- local pretrained model weights under `pretrained_models/`
- caches and IDE files

A good commit for this stage should describe the method change clearly, for example:
- `feat: add GPT-2 coarse prior imputation branch`
- `exp: switch from llm_info injection to x_prior-only`
