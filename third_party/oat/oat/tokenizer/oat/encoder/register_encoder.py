import torch
import torch.nn as nn
from functools import lru_cache

from oat.tokenizer.oat.model.sample_emb import SampleEmbedder
from oat.tokenizer.oat.model.pos_emb import PositionalEmbeddingAdder
from oat.tokenizer.oat.model.transformer import Transformer
from oat.tokenizer.oat.model.head import LinearHead


@lru_cache(maxsize=64)
def _create_causal_last_mask_cpu(action_len: int, register_len: int) -> torch.Tensor:
    """
    CPU-only boolean mask pattern for [actions, registers] attention.
    Cached tensors must not be tied to a specific CUDA context; `.to(device)` per forward.
    """
    total_len = action_len + register_len
    mask = torch.zeros((total_len, total_len), dtype=torch.bool)
    causal_reg_mask = torch.triu(
        torch.ones((register_len, register_len), dtype=torch.bool), diagonal=1
    )
    mask[action_len:, action_len:] = causal_reg_mask
    mask[:action_len, action_len:] = True
    return ~mask


def create_causal_last_mask(
    action_len: int,
    register_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Attention pattern:
    - Actions attend to all actions, not registers
    - Registers attend to all actions + causal among registers
    """
    return _create_causal_last_mask_cpu(action_len, register_len).to(device=device)


class RegisterEncoder(nn.Module):
    def __init__(self,
        # sample attrs
        sample_dim: int,
        sample_horizon: int,
        # encoder args
        emb_dim: int,
        head_dim: int,
        depth: int,
        pdropout: float,
        # latent args
        latent_dim: int,
        num_registers: int,
    ):
        super().__init__()
        
        self.sample_emb = SampleEmbedder(sample_dim, emb_dim)
        self.pos_emb = PositionalEmbeddingAdder(emb_dim, max_sizes=[sample_horizon,])
        self.registers = nn.Parameter(torch.randn(num_registers, emb_dim))
        self.transformer = Transformer(
            dim=emb_dim,
            depth=depth,
            head_dim=head_dim,
            drop=pdropout,
        )
        self.head = LinearHead(emb_dim, latent_dim)
        self.num_registers = num_registers

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        # sample: (B, T, sample_dim)
        # return latents: (B, num_registers, latent_dim)

        B, T, _ = sample.shape
        
        # proj & pos emb
        sample_emb = self.sample_emb(sample)    # (B, T, emb_dim)
        sample_emb = self.pos_emb(sample_emb)   # (B, T, emb_dim)

        # pad registers
        x = torch.cat([    # (B, T + num_registers, emb_dim)
            sample_emb,
            self.registers.unsqueeze(0).expand(B, -1, -1)
        ], dim=1)
        
        # transformer forward
        causal_last_attn_mask = create_causal_last_mask(T, self.num_registers, sample.device)
        x = self.transformer(x, block_mask=causal_last_attn_mask)
        
        # extract register tokens and project to latent space
        latents = self.head(x[:, T:])       # (B, num_registers, latent_dim)
        
        return latents
    