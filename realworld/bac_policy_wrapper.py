import sys
from pathlib import Path
import torch
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from BACInfer.core.diffusion_cache_wrapper import FastDiffusionPolicy

logger = logging.getLogger(__name__)


class BACPolicyWrapper:
    """BAC accelerated Policy wrapper class"""
    
    def __init__(self, policy, enable_bac=True, cache_config=None):
        self.original_policy = policy
        self.enable_bac = enable_bac
        self.cache_config = cache_config or {}
        
        if enable_bac and self.cache_config.get('mode') != 'original':
            self.policy = self._apply_bac_acceleration()
        else:
            self.policy = self.original_policy
    
    def _apply_bac_acceleration(self):
        mode = self.cache_config.get('mode', 'threshold')
        
        if mode == 'optimal':
            optimal_steps_dir = self.cache_config.get('optimal_steps_dir')
            
            return FastDiffusionPolicy.apply_cache(
                policy=self.original_policy,
                cache_mode='optimal',
                optimal_steps_dir=optimal_steps_dir,
                num_caches=self.cache_config.get('num_caches', 5),
                metric=self.cache_config.get('metric', 'cosine'),
                num_bu_blocks=self.cache_config.get('num_bu_blocks', 0)
            )
        elif mode == 'threshold':
            return FastDiffusionPolicy.apply_cache(
                policy=self.original_policy,
                cache_mode='threshold',
                cache_threshold=self.cache_config.get('threshold', 5),
                num_bu_blocks=self.cache_config.get('num_bu_blocks', 0)
            )
        else:
            return self.original_policy
    
    def predict_action(self, obs_dict):
        return self.policy.predict_action(obs_dict)
    
    def reset(self):
        self.policy.reset()
        
        if self.enable_bac and hasattr(self.policy, 'reset_cache'):
            self.policy.reset_cache()
        
        return self
    
    def eval(self):
        self.policy.eval()
        return self
    
    def to(self, device):
        self.policy.to(device)
        return self
    
    @property
    def n_obs_steps(self):
        return self.policy.n_obs_steps
    
    @property
    def n_action_steps(self):
        return self.policy.n_action_steps
    
    @property
    def num_inference_steps(self):
        return self.policy.num_inference_steps
    
    @num_inference_steps.setter
    def num_inference_steps(self, value):
        self.policy.num_inference_steps = value
