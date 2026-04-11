from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple

import dill
import hydra
import torch
from omegaconf import OmegaConf

from oat.model.common.module_attr_mixin import ModuleAttrMixin
from oat.model.common.normalizer import LinearNormalizer


def _resolve_tokenizer_ckpt_path(raw: str) -> Optional[Path]:
    """Resolve tokenizer checkpoint path (Hydra often stores relative paths)."""
    p = Path(raw).expanduser()
    if p.is_file():
        return p
    for root in (Path.cwd(), Path(__file__).resolve().parents[2]):
        cand = (root / raw).resolve()
        if cand.is_file():
            return cand
    return None


def _reload_action_tokenizer_from_cfg(policy: torch.nn.Module, cfg: OmegaConf) -> None:
    """
    Policies are often built with `OATTok.from_checkpoint(tok_path)` and then trained; the
    policy `.ckpt` still contains a full copy of `action_tokenizer` weights. If that copy ever
    diverges (partial load, stale run, optimizer edge cases), the tokenizer head can stay at its
    zero-init (`LinearHead` defaults) and collapse latents/tokens.

    Re-loading from the original tokenizer checkpoint restores the intended frozen tokenizer.
    """
    tok_cfg = OmegaConf.select(cfg, "policy.action_tokenizer")
    if tok_cfg is None:
        return
    ckpt = OmegaConf.select(tok_cfg, "checkpoint")
    if ckpt is None:
        return
    ckpt_s = str(ckpt).strip()
    if not ckpt_s or ckpt_s in ("???", "..."):
        return
    path = _resolve_tokenizer_ckpt_path(ckpt_s)
    if path is None:
        print(f"[warn] action_tokenizer.checkpoint not found on disk ({ckpt_s}); skip tokenizer refresh")
        return
    tok = getattr(policy, "action_tokenizer", None)
    if tok is None or not hasattr(tok.__class__, "from_checkpoint"):
        return
    ref, _ = tok.__class__.from_checkpoint(str(path), return_configuration=True)
    try:
        tok.load_state_dict(ref.state_dict(), strict=True)
    except Exception as exc:
        incomp = tok.load_state_dict(ref.state_dict(), strict=False)
        print(
            f"[warn] strict tokenizer reload failed ({exc}); "
            f"loaded non-strict. missing={incomp.missing_keys} unexpected={incomp.unexpected_keys}"
        )
    try:
        head = getattr(getattr(tok, "encoder", None), "head", None)
        proj = getattr(head, "proj", None)
        if proj is not None:
            wsum = float(proj.weight.detach().float().abs().sum())
            print(f"[oat] action_tokenizer refreshed from {path} (|encoder.head.proj|_1={wsum:.4f})")
        else:
            print(f"[oat] action_tokenizer refreshed from {path}")
    except Exception as exc:
        print(f"[oat] action_tokenizer refreshed from {path} (head stats skipped: {exc})")

class BasePolicy(ModuleAttrMixin):
    n_obs_steps: int
    n_action_steps: int

    @classmethod
    def from_checkpoint(cls, 
        checkpoint: str,
        output_dir: Optional[str] = None,
        return_configuration: bool = False,
    ):
        payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=output_dir, lazy_instantiation=False)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # Restore frozen tokenizer from the canonical OATTok (etc.) checkpoint after policy weights
        # are loaded — policy checkpoints can carry a broken copy while cfg still points at tok ckpt.
        _reload_action_tokenizer_from_cfg(workspace.model, cfg)
        em = getattr(workspace, "ema_model", None)
        if em is not None:
            _reload_action_tokenizer_from_cfg(em, cfg)

        policy = workspace.model
        if getattr(cfg.training, 'use_ema', False):
            policy = workspace.ema_model

        # Match training-time behavior in `TrainPolicyWorkspace.run()`:
        # fit normalizer from the task dataset and apply it to the policy.
        #
        # Without this, eval / ad-hoc diagnostics can silently use mismatched
        # normalization for observations/actions vs the zarr training distribution.
        dataset_cfg = OmegaConf.select(cfg, "task.policy.dataset")
        if dataset_cfg is not None:
            try:
                dataset = hydra.utils.instantiate(dataset_cfg)
                normalizer = dataset.get_normalizer()
                workspace.model.set_normalizer(normalizer)
                if getattr(cfg.training, "use_ema", False) and getattr(workspace, "ema_model", None) is not None:
                    workspace.ema_model.set_normalizer(normalizer)
            except Exception as exc:
                # Don't hard-fail checkpoint loading for non-standard checkpoints;
                # training entrypoints still set normalizers explicitly.
                print(f"[warn] Failed to apply dataset normalizer from cfg: {exc}")
        
        if return_configuration:
            return policy, cfg
        else:
            return policy
    
    def get_optimizer(self, *args, **kwargs):
        return torch.optim.AdamW(self.parameters(), *args, **kwargs)

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            str: B,To,*
        return: B,Ta,Da
        """
        raise NotImplementedError()

    def reset(self):
        pass

    def set_normalizer(self, normalizer: Union[LinearNormalizer, List[LinearNormalizer]]):
        raise NotImplementedError()
    
    def get_observation_encoder(self):
        raise NotImplementedError()
    
    def get_observation_modalities(self) -> List[str]:
        raise NotImplementedError()
    
    def get_observation_ports(self) -> List[str]:
        raise NotImplementedError()
    
    def get_policy_name(self) -> str:
        raise NotImplementedError()
    
    def create_dummy_observation(self,
        batch_size: int,
        horizon: int,
        obs_key_shapes: Dict[str, Tuple[int]],
        device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        obs_dict = dict()
        for obs_port, obs_shape in obs_key_shapes.items():
            obs_dict[obs_port] = torch.randn(
                size=(batch_size, horizon, *obs_shape),
            ).to(device)
        return obs_dict
