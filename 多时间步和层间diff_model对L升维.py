import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from linear_attention_transformer import LinearAttentionTransformer
except ImportError:
    LinearAttentionTransformer = None


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def get_linear_trans(heads=8, layers=1, channels=64, localheads=0, localwindow=0):
    if LinearAttentionTransformer is None:
        raise ImportError("linear_attention_transformer is required when diffusion.is_linear=true")
    return LinearAttentionTransformer(
        dim=channels,
        depth=layers,
        heads=heads,
        max_seq_len=256,
        n_local_attn_heads=0,
        local_attn_window_size=0,
    )


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim // 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)
        table = steps * frequencies
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, is_linear=False):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.is_linear = is_linear
        self.noise_proj1 = Conv1d_with_init(channels, channels, 1)
        self.noise_proj2 = Conv1d_with_init(channels, 1, 1)
        nn.init.zeros_(self.noise_proj2.weight)

        self.upsampler = nn.ConvTranspose2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, 2),
            stride=(1, 2),
            padding=(0, 0),
        )
        self.downsampler = nn.Conv2d(
            in_channels=2 * channels,
            out_channels=2 * channels,
            kernel_size=(1, 2),
            stride=(1, 2),
            padding=(0, 0),
        )

        if is_linear:
            self.time_layer = get_linear_trans(heads=nheads, layers=1, channels=channels)
            self.feature_layer = get_linear_trans(heads=nheads, layers=1, channels=channels)
        else:
            self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
            self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        if self.is_linear:
            y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        if self.is_linear:
            y = self.feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb, cumulative_skip, need_noise_pred=True):
        B, channel, K, L = x.shape
        original_shape = x.shape

        x_upsampled = self.upsampler(x)
        base_shape = x_upsampled.shape
        new_K, new_L = base_shape[2], base_shape[3]
        x_upsampled = x_upsampled.reshape(B, channel, new_K * new_L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)
        y = x_upsampled + diffusion_emb
        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)
        y = self.mid_projection(y)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)

        y_reshaped = y.reshape(B, 2 * channel, new_K, new_L)
        y_downsampled = self.downsampler(y_reshaped)
        y_downsampled = y_downsampled.reshape(B, 2 * channel, K * L)

        y = y_downsampled + cond_info
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(y)

        residual, current_skip = torch.chunk(y, 2, dim=1)
        residual = residual.reshape(original_shape)
        current_skip = current_skip.reshape(original_shape)
        cumulative_skip = cumulative_skip + current_skip

        noise_pred = None
        if need_noise_pred:
            noise_pred = cumulative_skip.reshape(B, channel, K * L)
            noise_pred = self.noise_proj1(noise_pred)
            noise_pred = F.relu(noise_pred)
            noise_pred = self.noise_proj2(noise_pred)
            noise_pred = noise_pred.reshape(B, K, L)

        return (x + residual) / math.sqrt(2.0), current_skip, cumulative_skip, noise_pred


class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.threshold = config.get("noise_threshold", 1e-5)
        self.current_cond_mask = None

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    is_linear=config["is_linear"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, diffusion_step, gt_noise=None, cond_mask=None):
        B, inputdim, K, L = x.shape
        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)
        diffusion_emb = self.diffusion_embedding(diffusion_step)

        cumulative_skip = torch.zeros_like(x)
        self.current_cond_mask = cond_mask.clone() if cond_mask is not None else None
        current_side_info = cond_info.clone()
        total_loss = 0.0
        need_layer = gt_noise is not None and cond_mask is not None
        layer_noise_preds = [] if need_layer else None

        for layer in self.residual_layers:
            x, current_skip, cumulative_skip, noise_pred = layer(
                x, current_side_info, diffusion_emb, cumulative_skip, need_noise_pred=need_layer
            )
            if need_layer:
                layer_noise_preds.append(noise_pred)

            if need_layer:
                target_mask = 1 - self.current_cond_mask
                layer_loss = ((noise_pred - gt_noise) ** 2 * target_mask).sum() / (target_mask.sum() + 1e-8)
                total_loss += layer_loss

                noise_error = torch.abs(noise_pred - gt_noise) * target_mask
                new_known_mask = (noise_error < self.threshold).float()
                self.current_cond_mask = torch.clamp(self.current_cond_mask + new_known_mask, 0, 1)

                if current_side_info.shape[1] > 0 and self.current_cond_mask is not None:
                    non_mask_part = current_side_info[:, :-1]
                    new_mask = self.current_cond_mask.unsqueeze(1)
                    current_side_info = torch.cat([non_mask_part, new_mask], dim=1)

        final_noise = self.output_projection1(cumulative_skip.reshape(B, self.channels, K * L))
        final_noise = F.relu(final_noise)
        final_noise = self.output_projection2(final_noise).reshape(B, K, L)
        if need_layer:
            layer_noise_preds.append(final_noise)

        if gt_noise is not None:
            return final_noise, total_loss, layer_noise_preds
        return final_noise, None, None
