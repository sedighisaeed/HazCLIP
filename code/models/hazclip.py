"""HazCLIP: Haze-aware Adapted CLIP (design v0.1, Method_Notes §2).

Architecture
------------
    Frozen CLIP visual encoder + hand-rolled LoRA (r=8) on Q/V projections
      |-- h_head (light MLP on CLS patch tokens) -> (beta_hat, t_stats) -> L_phys
      |-- weather-conditioned learnable prompt tokens + frozen text encoder
      `-- embedding e(x) = normalize(vision_out)  (prompt conditions the space)

Loss (train_hazclip.py)
-----------------------
    L = InfoNCE(cross-domain clean<->hazy) + lambda1 * L_phys + lambda2 * L_inv

No external PEFT library — LoRA is ~30 lines injected into the last N
attention blocks of CLIP-ViT (configurable), keeping the footprint tiny.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- LoRA layer

class LoRALinear(nn.Module):
    """Wrap a frozen nn.Linear with a low-rank residual: W x + (B A) x."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        d_out, d_in = base.weight.shape
        self.lora_A = nn.Parameter(torch.empty(r, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * (x @ self.lora_A.T @ self.lora_B.T)


# --------------------------------------------------------- injection helpers

def inject_lora_into_clip(clip_vision: nn.Module, r: int = 8,
                          last_n_blocks: int = 4) -> list[str]:
    """Replace q_proj/v_proj of the LAST `last_n_blocks` ViT blocks with LoRA.

    Works with transformers>=4.30 CLIPVisionEmbeddings/CLIPEncoderLayer layout.
    Returns the list of wrapped module paths (for the optimizer param group).
    """
    blocks = clip_vision.encoder.layers
    n = len(blocks)
    wrapped = []
    for bi in range(max(0, n - last_n_blocks), n):
        blk = blocks[bi]
        self_attn = blk.self_attn
        for name in ('q_proj', 'v_proj'):
            base = getattr(self_attn, name)
            setattr(self_attn, name, LoRALinear(base, r=r))
            wrapped.append(f'blocks[{bi}].self_attn.{name}')
    return wrapped


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


# ---------------------------------------------------------------- h_head

class HazeHead(nn.Module):
    """Light MLP over CLIP ViT features predicting haze physics:
    target = [beta, mean(t), std(t), min(t)] (4 scalars; see haze_synth)."""

    def __init__(self, d_in: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden), nn.GELU(),
            nn.Linear(hidden, 4),
            nn.Softplus(),          # all targets are non-negative
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: (B, N, d) patch/CLS-token features -> (B, 4)."""
        pooled = feats.mean(dim=1) if feats.dim() == 3 else feats
        return self.net(pooled)


# ------------------------------------------------------- weather prompts

WEATHER_TOKENS = {
    'clear': 'a photo of {}, clear weather',
    'light': 'a photo of {}, slightly hazy weather',
    'mild': 'a photo of {}, hazy weather',
    'moderate': 'a photo of {}, foggy weather with moderate visibility',
    'heavy': 'a photo of {}, dense fog with poor visibility',
    'extreme': 'a photo of {}, extremely dense haze, visibility near zero',
}
# Ordered severity index used to condition prompt embeddings
SEV2IDX = {s: i for i, s in enumerate(WEATHER_TOKENS)}


class WeatherPromptBank(nn.Module):
    """Learnable weather-conditioned prompt embedding offsets.

    For each of the 6 severities we learn a residual offset added to the
    frozen CLIP text embeddings of 'a photo of {class}, {weather} ...' prompts.
    This keeps class semantics (frozen text tower) while letting training
    pull all weather variants of a class toward one retrieval point.
    """

    def __init__(self, n_weathers: int = 6, d_model: int = 512):
        super().__init__()
        self.offsets = nn.Parameter(torch.zeros(n_weathers, d_model))
        nn.init.normal_(self.offsets, std=0.02)

    def forward(self, sev_idx: int | torch.Tensor) -> torch.Tensor:
        return self.offsets[sev_idx]


# ---------------------------------------------------------------- main model

class HazCLIP(nn.Module):
    """CLIP + LoRA + HazeHead + WeatherPromptBank.

    image_features(x) -> (B, d) L2-normalized
    text_features(class_prompts, sev) -> (C, d) with weather offset applied
    physics_head(x) -> (B, 4) [beta_hat, t_mean, t_std, t_min]
    """

    def __init__(self, clip_model_name: str = 'openai/clip-vit-base-patch32',
                 lora_r: int = 8, lora_last_n: int = 4, device: str = 'cpu'):
        super().__init__()
        from transformers import CLIPModel
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        # freeze everything first
        for p in self.clip.parameters():
            p.requires_grad_(False)
        self.lora_paths = inject_lora_into_clip(
            self.clip.vision_model, r=lora_r, last_n_blocks=lora_last_n)
        d_vision = self.clip.config.projection_dim          # 512 for B/32
        d_hidden_v = self.clip.vision_model.config.hidden_size  # 768
        self.haze_head = HazeHead(d_hidden_v)
        self.weather_bank = WeatherPromptBank(d_model=d_vision)
        self.logit_scale = self.clip.logit_scale.detach()
        self.device_str = device

    # ---- vision ------------------------------------------------------------
    def vision_features(self, pixel_values: torch.Tensor):
        """Return (projected, normalized image embeddings, raw hidden states)."""
        out = self.clip.vision_model(pixel_values=pixel_values)
        pooled = out.pooler_output if out.pooler_output is not None \
            else out.last_hidden_state[:, 0]
        proj = self.clip.visual_projection(pooled)
        return F.normalize(proj, dim=-1), out.last_hidden_state

    @torch.no_grad()
    def encode_images_eval(self, pixel_values: torch.Tensor) -> torch.Tensor:
        emb, _ = self.vision_features(pixel_values)
        return emb

    # ---- text --------------------------------------------------------------
    def text_features(self, input_ids: torch.Tensor,
                      attention_mask: torch.Tensor,
                      sev_idx: int | None = None) -> torch.Tensor:
        out = self.clip.text_model(input_ids=input_ids,
                                   attention_mask=attention_mask)
        pooled = out.pooler_output if out.pooler_output is not None \
            else out.last_hidden_state[
                torch.arange(out.last_hidden_state.shape[0]),
                input_ids.argmax(dim=-1)]
        proj = self.clip.text_projection(pooled)
        if sev_idx is not None:
            proj = proj + self.weather_bank(sev_idx)
        return F.normalize(proj, dim=-1)

    # ---- physics -----------------------------------------------------------
    def physics(self, pixel_values: torch.Tensor) -> torch.Tensor:
        _, hidden = self.vision_features(pixel_values)
        return self.haze_head(hidden)

    # ---- convenience -------------------------------------------------------
    def trainable_state_dict(self):
        """Only LoRA + heads (tiny checkpoint)."""
        sd = {}
        for name, p in self.named_parameters():
            if p.requires_grad:
                sd[name] = p
        return sd

    def load_trainable(self, path: str):
        sd = torch.load(path, map_location='cpu')
        missing, unexpected = self.load_state_dict(sd, strict=False)
        # frozen params are 'missing' by design; only warn on unexpected
        assert not unexpected, f'unexpected keys: {unexpected}'


# ---------------------------------------------------------------- unit check

if __name__ == '__main__':
    m = HazCLIP(device='cpu')
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in m.parameters())
    print(f'trainable params: {n_train:,} / {n_total:,} '
          f'({100 * n_train / n_total:.3f}%)')
    print('wrapped:', m.lora_paths)
    x = torch.randn(2, 3, 224, 224)
    emb, hidden = m.vision_features(x)
    print('emb', tuple(emb.shape), 'hidden', tuple(hidden.shape))
    beta_hat = m.physics(x)
    print('physics out', tuple(beta_hat.shape))
