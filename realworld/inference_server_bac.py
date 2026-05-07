import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from realworld.inference_server import DPInferenceServerSSH
from realworld.bac_policy_wrapper import BACPolicyWrapper
from server_config import (ENABLE_BAC, CACHE_MODE, CACHE_THRESHOLD, NUM_CACHES, CACHE_METRIC, NUM_BU_BLOCKS, OPTIMAL_STEPS_DIR)


class BACInferenceServer(DPInferenceServerSSH):
    """BAC Accelerated Inference Server"""
    
    def _load_model(self):
        super()._load_model()
        
        if ENABLE_BAC:
            cache_config = {
                'mode': CACHE_MODE,
                'threshold': CACHE_THRESHOLD,
                'num_caches': NUM_CACHES,
                'metric': CACHE_METRIC,
                'num_bu_blocks': NUM_BU_BLOCKS,
                'optimal_steps_dir': self._resolve_optimal_steps_dir()
            }
            
            self.policy = BACPolicyWrapper(policy=self.policy, enable_bac=True, cache_config=cache_config)
    
    def _resolve_optimal_steps_dir(self):
        if OPTIMAL_STEPS_DIR is not None:
            return OPTIMAL_STEPS_DIR
        
        project_root = Path(__file__).parent.parent
        assets_dir = project_root / "assets"
        
        if not assets_dir.exists():
            return None
        
        realworld_dir = assets_dir / "realworld_collected" / "optimal_steps" / CACHE_METRIC
        if realworld_dir.exists():
            return str(realworld_dir)
        
        task_dirs = list(assets_dir.glob("*/"))
        for task_dir in task_dirs:
            optimal_dir = task_dir / "optimal_steps" / CACHE_METRIC
            if optimal_dir.exists():
                return str(optimal_dir)
        
        return None


if __name__ == "__main__":
    server = BACInferenceServer()
    server.start()
