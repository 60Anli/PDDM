import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from 多时间步和层间diff_model对L升维 import diff_CSDI
from runtime_llm_conditioner import RuntimeLLMConditioner


class CSDI_base(nn.Module):
    def __init__(self, target_dim, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]

        self.current_epoch = 0  # 跟踪当前训练的epoch
        # 新增：层级和时间步掩码更新阈值
        self.layer_threshold = config.get("model", {}).get("layer_threshold", 0.001)
        self.step_threshold = config.get("model", {}).get("step_threshold", 0.001)
        self.num_next_steps = config.get("model", {}).get("num_next_steps", 1)
        self.llm_config = config.get("model", {}).get("llm", {})
        self.use_llm = bool(self.llm_config.get("enabled", False))
        self.llm_mode = str(self.llm_config.get("mode", "cache")).lower()
        self.use_runtime_llm = self.use_llm and self.llm_mode == "runtime"
        self.use_cached_llm = self.use_llm and self.llm_mode == "cache"
        self.prior_loss_weight = float(self.llm_config.get("prior_loss_weight", 0.1))
        self.mechanism_config = config.get("model", {}).get("mechanism", {})
        self.use_mechanism = bool(self.mechanism_config.get("enabled", False))
        self.mechanism_hidden_dim = int(self.mechanism_config.get("hidden_dim", 32))
        self.mechanism_local_window = int(self.mechanism_config.get("local_window", 5))
        self.mechanism_temperature = float(self.mechanism_config.get("temperature", 0.25))

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim,
            embedding_dim=self.emb_feature_dim
        )
        if self.use_runtime_llm:
            self.runtime_llm_conditioner = RuntimeLLMConditioner(self.llm_config)
            self.prior_residual_gate = nn.Parameter(torch.zeros(1))
        else:
            self.runtime_llm_conditioner = None
            self.prior_residual_gate = None

        if self.use_mechanism:
            self.step_mechanism_head = nn.Sequential(
                nn.Linear(5, self.mechanism_hidden_dim),
                nn.GELU(),
                nn.Linear(self.mechanism_hidden_dim, 1),
            )
            nn.init.zeros_(self.step_mechanism_head[-1].weight)
            nn.init.constant_(self.step_mechanism_head[-1].bias, -3.0)
        else:
            self.step_mechanism_head = None

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim
        config_diff["noise_threshold"] = self.layer_threshold
        config_diff["mechanism_enabled"] = self.use_mechanism
        config_diff["mechanism_hidden_dim"] = self.mechanism_hidden_dim
        config_diff["mechanism_local_window"] = self.mechanism_local_window
        config_diff["mechanism_temperature"] = self.mechanism_temperature  # 传递层级阈值到diffmodel

        input_dim = 1 if self.is_unconditional == True else 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        # 扩散模型参数
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5,
                config_diff["beta_end"] ** 0.5,
                self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"],
                config_diff["beta_end"],
                self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def time_embedding(self, pos, d_model=128):
        d_model = d_model if d_model % 2 == 0 else d_model - 1
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0,
            torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_block_mask(self, observed_mask, seed_prob=0.15, min_seq=12, max_seq=24, noise_prob=0.05):
        cond_mask = observed_mask.clone()
        batch_size, feature_dim, time_dim = observed_mask.shape

        for batch_idx in range(batch_size):
            sample_mask = observed_mask[batch_idx].clone()
            seed_mask = torch.rand(feature_dim, time_dim, device=observed_mask.device) < seed_prob

            for feature_idx in range(feature_dim):
                seed_idx = torch.nonzero(seed_mask[feature_idx], as_tuple=False).flatten()
                for start_idx in seed_idx.tolist():
                    if max_seq > min_seq:
                        span = int(np.random.randint(min_seq, max_seq + 1))
                    else:
                        span = int(min_seq)
                    end_idx = min(time_dim, start_idx + span)
                    sample_mask[feature_idx, start_idx:end_idx] = 0.0

            if noise_prob > 0:
                noise_mask = (torch.rand(feature_dim, time_dim, device=observed_mask.device) >= noise_prob).float()
                sample_mask = sample_mask * noise_mask

            sample_mask = torch.minimum(sample_mask, observed_mask[batch_idx])
            target_mask = observed_mask[batch_idx] - sample_mask
            if target_mask.sum() <= 0 and observed_mask[batch_idx].sum() > 1:
                observed_idx = torch.nonzero(observed_mask[batch_idx].reshape(-1) > 0.5, as_tuple=False).flatten()
                hide_idx = observed_idx[torch.randint(len(observed_idx), (1,), device=observed_idx.device)]
                flat_mask = sample_mask.reshape(-1)
                flat_mask[hide_idx] = 0.0
                sample_mask = flat_mask.reshape_as(sample_mask)

            cond_mask[batch_idx] = sample_mask

        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1]
        return cond_mask

    def get_test_pattern_mask(self, observed_mask, test_pattern_mask):
        return observed_mask * test_pattern_mask

    def _distance_features(self, cond_mask):
        B, K, L = cond_mask.shape
        time_idx = torch.arange(L, device=cond_mask.device).view(1, 1, L).expand(B, K, L)
        hard_mask = (cond_mask > 0.5).float()

        prev_index = torch.where(hard_mask > 0, time_idx, torch.zeros_like(time_idx))
        prev_index, _ = torch.cummax(prev_index, dim=-1)
        prev_valid = hard_mask.cumsum(dim=-1) > 0

        reversed_mask = torch.flip(hard_mask, dims=[-1])
        reversed_index = torch.flip(time_idx, dims=[-1])
        next_index_rev = torch.where(reversed_mask > 0, reversed_index, torch.zeros_like(reversed_index))
        next_index_rev, _ = torch.cummax(next_index_rev, dim=-1)
        next_index = torch.flip(next_index_rev, dims=[-1])
        next_valid = torch.flip(reversed_mask.cumsum(dim=-1) > 0, dims=[-1])

        denom = float(max(L - 1, 1))
        prev_dist = (time_idx - prev_index).float() / denom
        next_dist = (next_index - time_idx).float() / denom
        prev_dist = torch.where(prev_valid, prev_dist, torch.ones_like(prev_dist))
        next_dist = torch.where(next_valid, next_dist, torch.ones_like(next_dist))
        return prev_dist, next_dist

    def build_mechanism_features(self, cond_mask, error_map):
        target_mask = torch.clamp(1.0 - cond_mask, min=0.0, max=1.0)
        prev_dist, next_dist = self._distance_features(cond_mask)
        window = max(1, int(self.mechanism_local_window))
        if window % 2 == 0:
            window += 1
        local_missing = F.avg_pool1d(
            target_mask.reshape(-1, 1, target_mask.shape[-1]),
            kernel_size=window,
            stride=1,
            padding=window // 2,
        ).reshape_as(target_mask)
        effective_error = error_map * target_mask
        error_scale = effective_error.sum(dim=-1, keepdim=True) / target_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        error_norm = effective_error / error_scale.clamp_min(1e-6)
        features = torch.stack([cond_mask, prev_dist, next_dist, local_missing, error_norm], dim=-1)
        return features, target_mask

    def get_step_update_prob(self, cond_mask, error_map):
        target_mask = torch.clamp(1.0 - cond_mask, min=0.0, max=1.0)
        if not self.use_mechanism or self.step_mechanism_head is None:
            return (error_map < self.step_threshold).float() * target_mask
        features, target_mask = self.build_mechanism_features(cond_mask, error_map)
        mechanism_score = torch.sigmoid(self.step_mechanism_head(features).squeeze(-1))
        error_gate = torch.sigmoid((1.0 - features[..., -1]) / self.mechanism_temperature)
        return mechanism_score * error_gate * target_mask

    def get_x_prior(self, batch, observed_data=None, cond_mask=None):
        if not self.use_llm or not self.use_runtime_llm:
            return None
        if observed_data is None or cond_mask is None:
            return None
        outputs = self.runtime_llm_conditioner(observed_data, cond_mask)
        if isinstance(outputs, tuple):
            return outputs[-1]
        return outputs

    def get_external_cond_mask(self, batch):
        if "cond_mask" not in batch:
            return None
        return batch["cond_mask"].to(self.device).float().permute(0, 2, 1)

    def compute_prior_loss(self, observed_data, cond_mask, observed_mask, x_prior):
        if x_prior is None or self.prior_loss_weight <= 0:
            return torch.tensor(0.0, device=self.device)
        target_mask = torch.clamp(observed_mask - cond_mask, min=0, max=1)
        num_eval = target_mask.sum()
        if num_eval <= 0:
            return torch.tensor(0.0, device=self.device)
        residual = (x_prior - observed_data) * target_mask
        return (residual ** 2).sum() / num_eval

    def add_prior_loss(self, loss, observed_data, cond_mask, observed_mask, x_prior):
        if x_prior is None or self.prior_loss_weight <= 0:
            return loss
        prior_loss = self.compute_prior_loss(observed_data, cond_mask, observed_mask, x_prior)
        return loss + self.prior_loss_weight * prior_loss

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)
        side_info = side_info.permute(0, 3, 2, 1)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def calc_loss_valid(
            self, observed_data, cond_mask, observed_mask, side_info, x_prior, is_train
    ):
        loss_sum = 0
        for t in range(self.num_steps):
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, x_prior, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
            self, observed_data, cond_mask, observed_mask, side_info, x_prior, is_train, set_t=-1
    ):
        B, K, L = observed_data.shape
        original_target_mask = torch.clamp(observed_mask - cond_mask, min=0, max=1)
        original_num_eval = original_target_mask.sum()
        if original_num_eval == 0:
            return torch.tensor(0.0, device=self.device)

        if is_train != 1:  # 验证模式：单时间步+层级掩码更新
            t = (torch.ones(B) * set_t).long().to(self.device)
            current_alpha = self.alpha_torch[t]
            gt_noise = torch.randn_like(observed_data)
            noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * gt_noise
            total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask, x_prior=x_prior)

            # 调用diffmodel时传入gt_noise和cond_mask，启用层级更新
            predicted, layer_loss, _ = self.diffmodel(
                total_input, side_info, t, gt_noise=gt_noise, cond_mask=cond_mask
            )

            residual = (gt_noise - predicted) * original_target_mask
            loss = (residual ** 2).sum() / (original_num_eval if original_num_eval > 0 else 1)
            return loss + layer_loss  # 联合层级损失

        else:  # 训练模式：多步时间预测 + 层级动态更新
            # 1. 选择主时间步t和后续num_next_steps个时间步
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
            t = torch.clamp(t, max=self.num_steps - 1)

            next_steps_list = []
            for i in range(B):
                t_val = t[i].item()
                if t_val == 0:
                    next_steps = torch.tensor([], dtype=torch.long)
                else:
                    available_steps = min(self.num_next_steps, t_val)
                    next_steps = torch.randperm(t_val)[:available_steps] if available_steps > 0 else torch.tensor([],
                                                                                                                  dtype=torch.long)
                next_steps_list.append(next_steps)

            # 构建并排序时间步（从大到小）
            time_steps = [t]
            for i in range(self.num_next_steps):
                step_i = [next_steps_list[j][i].item() if i < len(next_steps_list[j]) else t[j].item() for j in
                          range(B)]
                time_steps.append(torch.tensor(step_i, device=self.device))

            first_batch_steps = [step[0].item() for step in time_steps]
            sorted_indices = sorted(range(len(first_batch_steps)), key=lambda i: first_batch_steps[i], reverse=True)
            time_steps = [time_steps[i] for i in sorted_indices]

            # 2. 初始化掩码和侧信息
            current_cond_mask = cond_mask.clone()
            current_side_info = side_info.clone()
            if not self.is_unconditional:
                current_side_info = torch.cat(
                    [current_side_info[:, :-1], current_cond_mask.unsqueeze(1)], dim=1
                )

            losses = []
            total_added_layer = []  # 层级更新新增掩码数
            total_added_step = []  # 时间步更新新增掩码数

            # 3. 按时间步处理（每个时间步内启用层级更新）
            for step in time_steps:
                step = torch.clamp(step, max=self.num_steps - 1)
                dynamic_target_mask = torch.clamp(observed_mask - current_cond_mask, min=0, max=1)
                current_num_eval = dynamic_target_mask.sum()
                if current_num_eval == 0:
                    losses.append((step, torch.tensor(0.0, device=self.device)))
                    continue

                # a. 生成带噪数据和真实噪声
                gt_noise = torch.randn_like(observed_data)
                alpha = self.alpha_torch[step]
                noisy_data = (alpha ** 0.5) * observed_data + (1.0 - alpha) ** 0.5 * gt_noise

                # b. 调用diffmodel，启用层级掩码更新
                total_input = self.set_input_to_diffmodel(noisy_data, observed_data, current_cond_mask, x_prior=x_prior)
                noise_pred, layer_loss, _ = self.diffmodel(
                    total_input, current_side_info, step,
                    gt_noise=gt_noise, cond_mask=current_cond_mask
                )

                # c. 计算该时间步损失（不含层级损失）
                residual = (gt_noise - noise_pred) * dynamic_target_mask
                step_loss = (residual ** 2).sum() / (current_num_eval if current_num_eval > 0 else 1)
                total_step_loss = step_loss # 联合损失
                losses.append((step, total_step_loss))

                # d. 提取层级更新后的掩码（从diffmodel中获取最终掩码）
                final_mask = self.diffmodel.current_cond_mask  # 关键：获取层级更新后的掩码
                added_layer = (final_mask.sum() - current_cond_mask.sum()).item()
                total_added_layer.append(added_layer)

                # e. 非主时间步：结合层级更新结果进一步更新掩码
                is_main_step = (step == t).all()
                if not is_main_step:
                    mae = torch.abs(gt_noise - noise_pred)
                    to_update = (mae < self.step_threshold).float() * dynamic_target_mask
                    to_update = to_update.detach()
                    added_step = to_update.sum().item()
                    total_added_step.append(added_step)

                    # 合并层级更新和时间步更新的掩码
                    current_cond_mask = torch.clamp(final_mask + to_update, 0, 1)
                    current_cond_mask = torch.min(current_cond_mask, observed_mask).detach()

                    # 更新侧信息
                    if not self.is_unconditional:
                        current_side_info = torch.cat(
                            [current_side_info[:, :-1], current_cond_mask.unsqueeze(1)], dim=1
                        ).detach()

            # 4. 损失加权
            main_loss_weight = 0.9
            other_loss_weight = (1 - main_loss_weight) / self.num_next_steps
            total_loss = 0.0
            main_step_val = t[0].item()

            loss_info = []
            for step, loss in losses:
                step_val = step[0].item()
                if step_val == main_step_val:
                    total_loss += main_loss_weight * loss
                    # loss_info.append(f"主步{step_val}: {loss.item():.6f}(权重{main_loss_weight})")
                else:
                    total_loss += other_loss_weight * loss
                    # loss_info.append(f"步{step_val}: {loss.item():.6f}(权重{other_loss_weight:.4f})")

            # # 打印日志：整合层级和时间步更新信息
            # print(
            #     f"训练阶段 | 主t={main_step_val} | 后续步数={self.num_next_steps} | "
            #     f"层级新增掩码={sum(total_added_layer):.0f} | 时间步新增掩码={sum(total_added_step):.0f} | " +
            #     " | ".join(loss_info)
            # )

            return total_loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask, x_prior=None):
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            if x_prior is not None and self.prior_residual_gate is not None:
                prior_target = ((1 - cond_mask) * x_prior).unsqueeze(1)
                noisy_target = noisy_target + torch.tanh(self.prior_residual_gate) * prior_target
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
        return total_input

    def impute(self, observed_data, cond_mask, side_info, x_prior, n_samples):
        B, K, L = observed_data.shape
        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional == True:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)
                else:
                    diff_input = self.set_input_to_diffmodel(current_sample, observed_data, cond_mask, x_prior=x_prior)
                # 推理阶段不更新掩码，仅预测
                predicted, _, _ = self.diffmodel(
                    diff_input, side_info, torch.tensor([t]).to(self.device),
                    gt_noise=None, cond_mask=cond_mask
                )

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = ((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
        ) = self.process_data(batch)
        external_cond_mask = self.get_external_cond_mask(batch)

        if is_train == 0:
            cond_mask = gt_mask
        elif self.use_cached_llm and external_cond_mask is not None:
            cond_mask = external_cond_mask
        elif self.target_strategy == "block":
            cond_mask = self.get_block_mask(observed_mask)
        elif self.target_strategy in {"mix", "historical"}:
            cond_mask = self.get_hist_mask(
                observed_mask, for_pattern_mask=for_pattern_mask
            )
        else:
            cond_mask = self.get_randmask(observed_mask)

        x_prior = self.get_x_prior(batch, observed_data, cond_mask)
        side_info = self.get_side_info(observed_tp, cond_mask)
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        loss = loss_func(observed_data, cond_mask, observed_mask, side_info, x_prior, is_train)
        return self.add_prior_loss(loss, observed_data, cond_mask, observed_mask, x_prior)

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask
            x_prior = self.get_x_prior(batch, observed_data, cond_mask)
            side_info = self.get_side_info(observed_tp, cond_mask)
            samples = self.impute(observed_data, cond_mask, side_info, x_prior, n_samples)

            for i in range(len(cut_length)):
                target_mask[i, ..., 0: cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp


class CSDI_SeqImputation(CSDI_base):
    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        for_pattern_mask = batch.get("hist_mask", batch["observed_mask"]).to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )


class CSDI_ETT(CSDI_SeqImputation):
    def __init__(self, config, device, target_dim=7):
        super(CSDI_ETT, self).__init__(target_dim, config, device)


class CSDI_Weather(CSDI_SeqImputation):
    def __init__(self, config, device, target_dim=21):
        super(CSDI_Weather, self).__init__(target_dim, config, device)


CSDI_weather = CSDI_Weather


class CSDI_PM25(CSDI_base):
    def __init__(self, config, device, target_dim=36):
        super(CSDI_PM25, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        for_pattern_mask = batch["hist_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )


class CSDI_Physio(CSDI_base):
    def __init__(self, config, device, target_dim=35):
        super(CSDI_Physio, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )


class CSDI_Forecasting(CSDI_base):
    def __init__(self, config, device, target_dim):
        super(CSDI_Forecasting, self).__init__(target_dim, config, device)
        self.target_dim_base = target_dim
        self.num_sample_features = config["model"]["num_sample_features"]

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        feature_id = torch.arange(self.target_dim_base).unsqueeze(0).expand(observed_data.shape[0], -1).to(self.device)
        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            feature_id,
        )

    def sample_features(self, observed_data, observed_mask, feature_id, gt_mask):
        size = self.num_sample_features
        self.target_dim = size
        extracted_data = []
        extracted_mask = []
        extracted_feature_id = []
        extracted_gt_mask = []

        for k in range(len(observed_data)):
            ind = np.arange(self.target_dim_base)
            np.random.shuffle(ind)
            extracted_data.append(observed_data[k, ind[:size]])
            extracted_mask.append(observed_mask[k, ind[:size]])
            extracted_feature_id.append(feature_id[k, ind[:size]])
            extracted_gt_mask.append(gt_mask[k, ind[:size]])
        extracted_data = torch.stack(extracted_data, 0)
        extracted_mask = torch.stack(extracted_mask, 0)
        extracted_feature_id = torch.stack(extracted_feature_id, 0)
        extracted_gt_mask = torch.stack(extracted_gt_mask, 0)
        return extracted_data, extracted_mask, extracted_feature_id, extracted_gt_mask

    def sample_cond_mask(self, cond_mask, feature_id):
        if cond_mask is None or feature_id is None or cond_mask.shape[1] == feature_id.shape[1]:
            return cond_mask
        gather_index = feature_id.unsqueeze(-1).expand(-1, -1, cond_mask.shape[-1])
        return torch.gather(cond_mask, 1, gather_index)

    def sample_llm_embedding(self, llm_embedding, feature_id):
        if llm_embedding is None or feature_id is None or llm_embedding.shape[1] == feature_id.shape[1]:
            return llm_embedding
        gather_index = feature_id.unsqueeze(-1).expand(-1, -1, llm_embedding.shape[-1])
        return torch.gather(llm_embedding, 1, gather_index)

    def build_forecasting_cond_mask(self, observed_mask, gt_mask):
        return self.get_test_pattern_mask(observed_mask, gt_mask)

    def get_side_info(self, observed_tp, cond_mask, feature_id=None):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, self.target_dim, -1)

        if self.target_dim == self.target_dim_base:
            feature_embed = self.embed_layer(torch.arange(self.target_dim).to(self.device))
            feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        else:
            feature_embed = self.embed_layer(feature_id).unsqueeze(1).expand(-1, L, -1, -1)
        side_info = torch.cat([time_embed, feature_embed], dim=-1)
        side_info = side_info.permute(0, 3, 2, 1)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)
            side_info = torch.cat([side_info, side_mask], dim=1)
        return side_info

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            _,
            feature_id,
        ) = self.process_data(batch)
        external_cond_mask = self.get_external_cond_mask(batch)

        self.target_dim = self.target_dim_base
        if is_train == 1 and (self.target_dim_base > self.num_sample_features):
            observed_data, observed_mask, feature_id, gt_mask = self.sample_features(
                observed_data, observed_mask, feature_id, gt_mask
            )
        else:
            self.target_dim = self.target_dim_base
            feature_id = None

        if self.use_cached_llm and external_cond_mask is not None:
            cond_mask = self.sample_cond_mask(external_cond_mask, feature_id)
        else:
            cond_mask = self.build_forecasting_cond_mask(observed_mask, gt_mask)

        x_prior = self.get_x_prior(batch, observed_data, cond_mask)
        if x_prior is not None and feature_id is not None and x_prior.shape[1] != feature_id.shape[1]:
            x_prior = self.sample_cond_mask(x_prior, feature_id)
        side_info = self.get_side_info(observed_tp, cond_mask, feature_id)
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        loss = loss_func(observed_data, cond_mask, observed_mask, side_info, x_prior, is_train)
        return self.add_prior_loss(loss, observed_data, cond_mask, observed_mask, x_prior)

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            _,
            feature_id,
        ) = self.process_data(batch)
        external_cond_mask = self.get_external_cond_mask(batch)

        with torch.no_grad():
            self.target_dim = self.target_dim_base
            if self.use_cached_llm and external_cond_mask is not None:
                cond_mask = external_cond_mask
            else:
                cond_mask = self.build_forecasting_cond_mask(observed_mask, gt_mask)
            target_mask = observed_mask * (1 - gt_mask)
            x_prior = self.get_x_prior(batch, observed_data, cond_mask)
            side_info = self.get_side_info(observed_tp, cond_mask, feature_id=None)
            samples = self.impute(observed_data, cond_mask, side_info, x_prior, n_samples)

        return samples, observed_data, target_mask, observed_mask, observed_tp