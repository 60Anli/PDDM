# 实验模式说明

这个文件专门说明仓库里不同配置模式的用途、区别，以及对应的运行方式。

## 1. 原始基线

对应配置：
- `config/base.yaml`

适用场景：
- 需要原始 PDDM 结果
- 所有新方法都应该先和它对比

常用命令：
```powershell
python exe_physio.py --config base.yaml --device cuda:0 --seed 1 --nfold 0 --testmissingratio 0.1 --nsample 100
python exe_pm25.py --config base.yaml --device cuda:0 --validationindex 0 --targetstrategy random --nsample 100
```

## 2. GPT-2 粗插补先验模式

对应配置：
- `config/base_llm.yaml`

适用场景：
- 想测试 GPT-2 粗插补先验路线
- 本地已经准备好 GPT-2 权重

说明：
- 这条路线依赖 `pretrained_models/gpt2`
- 更适合作为 LLM 对照实验或扩展实验

## 3. 缺失机制软更新模式

对应配置：
- `config/base_mechanism.yaml`

适用场景：
- 想复现 mechanism-aware mask-update 的原始思路
- 想和修复版作对比

说明：
- 这版可能不稳定
- 不建议作为第一优先长跑版本

## 4. 缺失机制硬阈值修复版

对应配置：
- `config/base_mechanism_hard.yaml`

适用场景：
- 想测试更稳的 mechanism 路线
- 想保留 mechanism encoder，但不让 soft mask 传播

行为特点：
- 保留 mechanism encoder
- step-level 更新改回硬二值更新
- 默认关闭 layer-level mechanism update

推荐命令：
```powershell
python exe_physio.py --config base_mechanism_hard.yaml --device cuda:0 --seed 1 --nfold 0 --testmissingratio 0.1 --nsample 100
python exe_pm25.py --config base_mechanism_hard.yaml --device cuda:0 --validationindex 0 --targetstrategy random --nsample 100
python exe_weather.py --config base_mechanism_hard.yaml --device cuda:0 --data_path ./data/Weather --eval_length 24 --missing_ratio 0.2 --missing_pattern random --target_strategy random --nsample 50
python exe_ettm1.py --config base_mechanism_hard.yaml --device cuda:0 --data_path ./data/ETT_processed/ETTm1 --raw_data_path ./data/ETT_raw/ETTm1.csv --eval_length 24 --missing_ratio 0.2 --missing_pattern random --target_strategy random --nsample 50
```

## 5. 周期 / 频域先验完整版

对应配置：
- `config/base_periodic.yaml`

适用场景：
- 想在不改 `cond_mask` 的前提下，引入周期结构先验
- 想同时测试时域先验和频域辅助损失

行为特点：
- 均值填充后做 FFT
- 保留 top-k 主要频率成分
- 同时使用时域损失和频域损失

## 6. 周期先验无频域损失版

对应配置：
- `config/base_periodic_nofreq.yaml`

适用场景：
- 想先判断“周期先验本身有没有帮助”
- 想排除频域辅助损失的影响

行为特点：
- 均值填充后做 FFT
- 不加频域辅助损失

推荐命令：
```powershell
python exe_physio.py --config base_periodic_nofreq.yaml --device cuda:0 --seed 1 --nfold 0 --testmissingratio 0.1 --nsample 100
python exe_pm25.py --config base_periodic_nofreq.yaml --device cuda:0 --validationindex 0 --targetstrategy random --nsample 100
python exe_weather.py --config base_periodic_nofreq.yaml --device cuda:0 --data_path ./data/Weather --eval_length 24 --missing_ratio 0.2 --missing_pattern random --target_strategy random --nsample 50
python exe_ettm1.py --config base_periodic_nofreq.yaml --device cuda:0 --data_path ./data/ETT_processed/ETTm1 --raw_data_path ./data/ETT_raw/ETTm1.csv --eval_length 24 --missing_ratio 0.2 --missing_pattern random --target_strategy random --nsample 50
```

## 7. 周期先验线性插值版

对应配置：
- `config/base_periodic_linear.yaml`

适用场景：
- 想比较“均值填充 vs 线性插值”哪种更适合提取周期先验
- 想看更强时域填充是否更利于频域建模

行为特点：
- 线性插值后再做 FFT
- 保留频域辅助损失
- 一般会比均值填充版更慢一些

推荐命令：
```powershell
python exe_physio.py --config base_periodic_linear.yaml --device cuda:0 --seed 1 --nfold 0 --testmissingratio 0.1 --nsample 100
python exe_pm25.py --config base_periodic_linear.yaml --device cuda:0 --validationindex 0 --targetstrategy random --nsample 100
python exe_weather.py --config base_periodic_linear.yaml --device cuda:0 --data_path ./data/Weather --eval_length 24 --missing_ratio 0.2 --missing_pattern random --target_strategy random --nsample 50
python exe_ettm1.py --config base_periodic_linear.yaml --device cuda:0 --data_path ./data/ETT_processed/ETTm1 --raw_data_path ./data/ETT_raw/ETTm1.csv --eval_length 24 --missing_ratio 0.2 --missing_pattern random --target_strategy random --nsample 50
```

## 建议比较顺序

如果你想更高效地比较最近几条路线，建议顺序：
1. `base.yaml`
2. `base_mechanism_hard.yaml`
3. `base_periodic_nofreq.yaml`
4. `base_periodic_linear.yaml`
5. `base_periodic.yaml`

这样最容易分清：
- 基线本身表现
- mechanism 修复版有没有帮助
- 周期先验本身有没有帮助
- 频域损失是否在帮忙
- 线性插值是否优于均值填充
