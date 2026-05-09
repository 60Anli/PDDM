import torch
import torch.nn as nn
import torch.nn.functional as F


class RuntimeLLMConditioner(nn.Module):
    """Mask-aware online LLM conditioner that predicts a residual prior over x_base."""

    def __init__(self, config):
        super().__init__()
        self.d_llm = int(config.get("embedding_dim", 768))
        self.proj_dim = int(config.get("proj_dim", 32))
        self.prefix_tokens = int(config.get("prefix_tokens", 4))
        self.patch_len = max(1, int(config.get("patch_len", 8)))
        self.dropout = float(config.get("dropout", 0.1))
        self.use_backbone = bool(config.get("use_backbone", True))
        self.fallback_to_transformer = bool(config.get("fallback_to_transformer", True))

        self.patch_projection = nn.Sequential(
            nn.Linear(self.patch_len * 3 + 6, self.d_llm),
            nn.GELU(),
            nn.Linear(self.d_llm, self.d_llm),
        )
        self.global_stat_projection = nn.Sequential(
            nn.Linear(6, self.d_llm),
            nn.GELU(),
            nn.Linear(self.d_llm, self.d_llm),
        )
        self.prefix = nn.Parameter(torch.zeros(1, self.prefix_tokens, self.d_llm))
        self.output_projection = nn.Sequential(
            nn.LayerNorm(self.d_llm),
            nn.Linear(self.d_llm, self.proj_dim),
            nn.SiLU(),
            nn.Linear(self.proj_dim, self.proj_dim),
        )
        self.prior_head = nn.Sequential(
            nn.LayerNorm(self.proj_dim),
            nn.Linear(self.proj_dim, self.proj_dim),
            nn.GELU(),
            nn.Linear(self.proj_dim, 1),
        )
        self.confidence_head = nn.Sequential(
            nn.LayerNorm(self.proj_dim),
            nn.Linear(self.proj_dim, self.proj_dim),
            nn.GELU(),
            nn.Linear(self.proj_dim, 1),
        )
        nn.init.zeros_(self.prior_head[-1].weight)
        nn.init.zeros_(self.prior_head[-1].bias)
        nn.init.zeros_(self.confidence_head[-1].weight)
        nn.init.zeros_(self.confidence_head[-1].bias)

        self.dropout_layer = nn.Dropout(self.dropout)
        self.backbone = self._build_backbone(config)

    def _build_backbone(self, config):
        if not self.use_backbone:
            return self._fallback_transformer(config)

        model_name = config.get("model_name", "gpt2")
        llm_layers = int(config.get("layers", 3))
        local_files_only = bool(config.get("local_files_only", False))
        strict = bool(config.get("strict_load", False))

        try:
            from transformers import GPT2Config, GPT2Model

            gpt_config = GPT2Config.from_pretrained(model_name)
            gpt_config.num_hidden_layers = llm_layers
            gpt_config.output_hidden_states = False
            gpt_config.output_attentions = False
            backbone = GPT2Model.from_pretrained(
                model_name,
                config=gpt_config,
                local_files_only=local_files_only,
            )
            if hasattr(backbone, "h"):
                backbone.h = backbone.h[:llm_layers]
            for param in backbone.parameters():
                param.requires_grad = False
            backbone.eval()
            return backbone
        except Exception as exc:
            if strict or not self.fallback_to_transformer:
                raise RuntimeError(f"Failed to load GPT-2 backbone: {exc}") from exc
            print(f"Warning: failed to load GPT-2 backbone, using Transformer fallback. {exc}")
            return self._fallback_transformer(config)

    def _fallback_transformer(self, config):
        heads = int(config.get("n_heads", 8))
        layers = int(config.get("layers", 3))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_llm,
            nhead=heads,
            dim_feedforward=4 * self.d_llm,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        return nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def _global_stats(self, observed_data, cond_mask):
        known = observed_data * cond_mask
        count = cond_mask.sum(dim=-1).clamp_min(1.0)
        mean = known.sum(dim=-1) / count
        centered = (observed_data - mean.unsqueeze(-1)) * cond_mask
        std = torch.sqrt((centered.square().sum(dim=-1) / count).clamp_min(1e-6))
        missing_rate = 1.0 - cond_mask.mean(dim=-1)

        min_values = observed_data.masked_fill(cond_mask == 0, float("inf")).amin(dim=-1)
        max_values = observed_data.masked_fill(cond_mask == 0, float("-inf")).amax(dim=-1)
        min_values = torch.where(torch.isfinite(min_values), min_values, torch.zeros_like(min_values))
        max_values = torch.where(torch.isfinite(max_values), max_values, torch.zeros_like(max_values))

        valid_diff = cond_mask[..., 1:] * cond_mask[..., :-1]
        diff_count = valid_diff.sum(dim=-1).clamp_min(1.0)
        trend = ((observed_data[..., 1:] - observed_data[..., :-1]) * valid_diff).sum(dim=-1) / diff_count

        return torch.stack([mean, std, min_values, max_values, trend, missing_rate], dim=-1)

    def _patchify(self, tensor):
        batch_size, feature_dim, time_dim = tensor.shape
        num_patches = (time_dim + self.patch_len - 1) // self.patch_len
        pad_len = num_patches * self.patch_len - time_dim
        if pad_len > 0:
            tensor = F.pad(tensor, (0, pad_len))
        return tensor.reshape(batch_size, feature_dim, num_patches, self.patch_len)

    def _patch_stats(self, patch_values, patch_mask):
        count = patch_mask.sum(dim=-1).clamp_min(1.0)
        mean = patch_values.sum(dim=-1) / count
        centered = (patch_values - mean.unsqueeze(-1)) * patch_mask
        std = torch.sqrt((centered.square().sum(dim=-1) / count).clamp_min(1e-6))
        missing_rate = 1.0 - patch_mask.mean(dim=-1)

        min_values = patch_values.masked_fill(patch_mask == 0, float("inf")).amin(dim=-1)
        max_values = patch_values.masked_fill(patch_mask == 0, float("-inf")).amax(dim=-1)
        min_values = torch.where(torch.isfinite(min_values), min_values, torch.zeros_like(min_values))
        max_values = torch.where(torch.isfinite(max_values), max_values, torch.zeros_like(max_values))

        valid_diff = patch_mask[..., 1:] * patch_mask[..., :-1]
        diff_count = valid_diff.sum(dim=-1).clamp_min(1.0)
        trend = ((patch_values[..., 1:] - patch_values[..., :-1]) * valid_diff).sum(dim=-1) / diff_count

        return torch.stack([mean, std, min_values, max_values, trend, missing_rate], dim=-1)

    def _restore_time_grid(self, patch_hidden, target_length):
        time_hidden = patch_hidden.repeat_interleave(self.patch_len, dim=2)
        return time_hidden[:, :, :target_length, :]

    def forward(self, observed_data, cond_mask, x_base=None):
        batch_size, feature_dim, time_dim = observed_data.shape
        known_values = observed_data * cond_mask
        if x_base is None:
            x_base = known_values

        patch_values = self._patchify(known_values)
        patch_mask = self._patchify(cond_mask)
        patch_base = self._patchify(x_base)
        patch_stats = self._patch_stats(patch_values, patch_mask)

        patch_inputs = torch.cat([patch_values, patch_mask, patch_base, patch_stats], dim=-1)
        patch_tokens = self.patch_projection(patch_inputs)
        global_tokens = self.global_stat_projection(self._global_stats(observed_data, cond_mask)).unsqueeze(2)
        variable_tokens = self.dropout_layer(patch_tokens + global_tokens)

        num_patches = variable_tokens.shape[2]
        llm_input = variable_tokens.reshape(batch_size * feature_dim, num_patches, self.d_llm)
        prefix = self.prefix.expand(batch_size * feature_dim, -1, -1)
        llm_input = torch.cat([prefix, llm_input], dim=1)

        if isinstance(self.backbone, nn.TransformerEncoder):
            hidden = self.backbone(llm_input)
        else:
            hidden = self.backbone(inputs_embeds=llm_input).last_hidden_state

        patch_hidden = hidden[:, -num_patches:, :]
        patch_hidden = self.output_projection(patch_hidden)
        patch_hidden = patch_hidden.reshape(batch_size, feature_dim, num_patches, self.proj_dim)
        time_hidden = self._restore_time_grid(patch_hidden, time_dim)
        residual_prior = self.prior_head(time_hidden).squeeze(-1)
        prior_confidence = torch.sigmoid(self.confidence_head(time_hidden).squeeze(-1))
        return residual_prior, prior_confidence
