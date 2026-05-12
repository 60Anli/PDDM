# 最近改动说明（2026-05-12）

这个文件用于概括最近在原始 PDDM 基线之上尝试过的几条主要路线，以及每条路线的目标。

## 1. GPT-2 粗插补先验路线

对应配置：
- `config/base_llm.yaml`

核心想法：
- 用 GPT-2 生成粗插补先验 `x_prior`
- 只在 target 位置通过残差门控注入 `x_prior`
- 最终插补仍由 diffusion 模型完成

目前结论：
- 已尝试过多种 LLM 接入方式
- 当前实验下，增益在不同数据集上不稳定

## 2. 缺失机制感知更新路线

对应配置：
- `config/base_mechanism.yaml`
- `config/base_mechanism_tune_*.yaml`

核心想法：
- 使用缺失模式相关特征，引导逐层 / 多时间步更新
- 尝试用机制感知的更新分数，替代完全固定的更新方式

目前结论：
- 软更新版本不稳定，可能明显拉坏性能
- 更适合作为探索性尝试，而不是当前默认主线

## 3. 缺失机制硬阈值修复版

对应配置：
- `config/base_mechanism_hard.yaml`

核心想法：
- 保留 mechanism encoder
- step-level 更新恢复成硬阈值二值更新
- 默认关闭 layer-level mechanism update
- 避免 soft probability 直接传播到 `cond_mask`

存在意义：
- 用来验证前一版的主要问题是不是来自 soft mask 传播
- 是 mechanism 路线的更稳修复版

## 4. 周期 / 频域先验路线

对应配置：
- `config/base_periodic.yaml`
- `config/base_periodic_nofreq.yaml`
- `config/base_periodic_linear.yaml`

核心想法：
- 不修改 `cond_mask`
- 直接在已有的 `x_prior` 通路上注入周期结构先验
- 通过 FFT 保留主要频率成分，再反变换回时域
- 可选加入频域辅助损失

不同版本：
- `base_periodic.yaml`
  均值填充后提取周期先验，并加频域损失
- `base_periodic_nofreq.yaml`
  均值填充后提取周期先验，但不加频域损失
- `base_periodic_linear.yaml`
  线性插值后提取周期先验，并加频域损失

存在意义：
- 这条路线不改任务掩码定义
- 比 mechanism 动态改 mask 的风险更低
- 更适合验证周期结构本身是否能帮助插补

## 推荐比较顺序

如果想更高效地做对比，建议顺序：
1. 原始 PDDM 基线：`base.yaml`
2. 缺失机制硬修复版：`base_mechanism_hard.yaml`
3. 周期先验无频域损失：`base_periodic_nofreq.yaml`
4. 周期先验线性插值版：`base_periodic_linear.yaml`
5. 周期先验完整版：`base_periodic.yaml`

## 实用建议

- soft mechanism 路线更适合当探索性负结果，不建议作为当前主线。
- 如果你想测试“周期先验本身有没有帮助”，先跑 `base_periodic_nofreq.yaml`。
- 如果你想比较“均值填充 vs 线性插值”，就对比 `base_periodic_nofreq.yaml` 和 `base_periodic_linear.yaml`。
