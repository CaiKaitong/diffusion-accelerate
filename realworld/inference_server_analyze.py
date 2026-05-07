import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from realworld.inference_server import DPInferenceServerSSH
from realworld.bac_realworld_analyzer import BACRealworldAnalyzer
from realworld.compute_bu_blocks import compute_and_save_bu_blocks
from server_config import CHECKPOINT_PATH, CACHE_METRIC


class AnalyzeInferenceServer(DPInferenceServerSSH):
    """Inference Server with Activation Collection"""
    
    def __init__(self, 
                 output_dir: str = "assets/realworld_collected",
                 num_caches_list: list = [5, 8, 10],
                 target_iteration: int = 100,
                 num_bu_blocks: int = 3,
                 **kwargs):
        self.analyze_output_dir = output_dir
        self.num_caches_list = num_caches_list
        self.target_iteration = target_iteration
        self.num_bu_blocks = num_bu_blocks
        self.analyzer = None
        self.iteration_count = 0
        self.collection_done = False
        
        super().__init__(**kwargs)
    
    def _load_model(self):
        super()._load_model()
        
        self.analyzer = BACRealworldAnalyzer(
            policy=self.policy,
            output_dir=self.analyze_output_dir,
            metric=CACHE_METRIC
        )
        
    
    def _infer_action(self, env_obs, timestamps):
        self.iteration_count += 1
        
        if self.iteration_count == self.target_iteration and not self.collection_done:
            if self.analyzer.activations:
                self.analyzer.clear_activations()
            
            self.analyzer.start_collecting()
            
            action = super()._infer_action(env_obs, timestamps)
            
            self.analyzer.stop_collecting()
            
            self._finish_collection()
            
            return action
        else:
            return super()._infer_action(env_obs, timestamps)
    
    def _finish_collection(self):
        filename = f"activations_iter_{self.iteration_count}.pkl"
        activations_file = self.analyzer.save_activations(filename)
        optimal_steps_all = self.analyzer.compute_optimal_steps(self.num_caches_list)
        
        steps_dir = self.analyzer.save_optimal_steps(
            optimal_steps_all,
            filename_prefix="optimal_steps"
        )
        
        if self.num_bu_blocks > 0:
            bu_dir = compute_and_save_bu_blocks(
                activations_path=str(activations_file),
                num_blocks=self.num_bu_blocks,
                output_dir=self.analyze_output_dir
            )
        
        self.collection_done = True
        
        self.analyzer.clear_activations()
    
    def _handle_client(self, client_socket, client_addr):
        super()._handle_client(client_socket, client_addr)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='BAC Activation Collection Server (Real Robot)')
    parser.add_argument('-o', '--output', default='assets/realworld_collected',
                       help='Output directory (default: assets/realworld_collected)')
    parser.add_argument('--num_caches', default='5,8,10,20',
                       help='Cache numbers, comma separated (default: 5,8,10)')
    parser.add_argument('--iteration', type=int, default=100,
                       help='Iteration to collect activations (default: 100)')
    parser.add_argument('--num_bu_blocks', type=int, default=3,
                       help='Number of BU blocks, 0 means skip (default: 3)')
    
    args = parser.parse_args()
    
    num_caches_list = [int(x.strip()) for x in args.num_caches.split(',')]
    
    server = AnalyzeInferenceServer(
        output_dir=args.output,
        num_caches_list=num_caches_list,
        target_iteration=args.iteration,
        num_bu_blocks=args.num_bu_blocks
    )
    
    server.start()
