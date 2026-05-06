# PDDM LLM Conditioning

The recommended path is now `model.llm.mode: runtime`, inspired by CALF and
Time-LLM. It does not cache one prompt per `cond_mask`. PDDM keeps its random
mask generation, then sends `observed_data * cond_mask` plus `cond_mask` through
a frozen GPT-2-style conditioner using `inputs_embeds`.

## Runtime LLM Training

Physio:

```bash
python exe_physio.py --config base_llm.yaml --device cuda:0 --seed 1 --nfold 0 --testmissingratio 0.1
```

PM25:

```bash
python exe_pm25.py --config base_llm.yaml --device cuda:0 --validationindex 0 --targetstrategy mix
```

## Optional Cache Mode

The older TimeCMA-style cache path is still available if you set:

```yaml
model:
  llm:
    enabled: true
    mode: "cache"
```

Then generate cache files before training:

```bash
python generate_llm_embeddings.py --dataset physio --config base_llm.yaml --split all --backend gpt2 --device cuda:0 --seed 1 --nfold 0 --testmissingratio 0.1
python generate_llm_embeddings.py --dataset pm25 --config base_llm.yaml --split all --backend gpt2 --device cuda:0 --validationindex 0 --targetstrategy mix
```

## Notes

- Runtime mode preserves mask randomness and avoids prompt-cache explosion.
- GPT-2 is frozen; only the PDDM model and reprogramming/projection layers train.
- The runtime LLM branch now predicts only a coarse `x_prior`, which is injected into target positions through a zero-initialized residual gate.
- `base_llm.yaml` forces GPT-2 from `./pretrained_models/gpt2`: `local_files_only: true`, `strict_load: true`, and `fallback_to_transformer: false`. If local GPT-2 cannot be loaded, training stops instead of using a small Transformer.
- Mixup is disabled by default because LLM conditioning is mask-aware.
