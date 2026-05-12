# PDDM 实验仓库说明

这个仓库是在原始 PDDM 插补框架上不断做实验得到的版本集合，重点保留了不同思路的配置和入口，方便直接对比。

## 基线

原始基线仍然是 PDDM 本身，核心脚本包括：
- `exe_physio.py`
- `exe_pm25.py`
- `exe_weather.py`
- `exe_ettm1.py`
- `多时间步和层间main_model.py`
- `多时间步和层间diff_model对L升维.py`

## 当前保留的实验方向

详细模式说明见：
- [EXPERIMENT_MODES.md](./EXPERIMENT_MODES.md)

当前主要包括：
- `base_llm.yaml`：GPT-2 粗插补先验路线
- `base_mechanism.yaml`：缺失机制软更新路线
- `base_mechanism_hard.yaml`：缺失机制硬阈值修复版
- `base_periodic.yaml`：均值填充 + 周期先验 + 频域损失
- `base_periodic_nofreq.yaml`：均值填充 + 周期先验，不加频域损失
- `base_periodic_linear.yaml`：线性插值 + 周期先验 + 频域损失

## 最近改动说明

简要改动总结见：
- [CHANGE_NOTES_20260512.md](./CHANGE_NOTES_20260512.md)

## 环境建议

推荐环境：
- conda 环境：`torch230cuda121`
- PyTorch
- pandas
- pyyaml
- tqdm
- transformers（只有 LLM 路线需要）

## 额外说明

- LLM 路线需要 `./pretrained_models/gpt2`
- mechanism 和 periodic 路线不依赖 GPT-2
- 仓库里还保留了 forecasting 相关脚本，但最近主要做的是插补实验
