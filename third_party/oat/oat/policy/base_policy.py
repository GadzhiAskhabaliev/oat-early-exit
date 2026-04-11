from typing import Dict, List, Union, Optional, Tuple
import torch
import dill
import hydra
from omegaconf import OmegaConf
from oat.model.common.module_attr_mixin import ModuleAttrMixin
from oat.model.common.normalizer import LinearNormalizer

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
