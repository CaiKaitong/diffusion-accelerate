import numpy as np
import torch
import logging
import pickle
from pathlib import Path
import argparse
import sys
from typing import List, Dict, Tuple, Optional
import json

logger = logging.getLogger(__name__)

class OptimalCacheScheduler:
    
    def __init__(self, similarity_matrix: np.ndarray = None, similarity_matrices_dir: str = None, 
                 module_name: str = None, metric: str = 'cosine'):
        self.similarity_matrix = None
        self.metric = metric
        self.optimal_steps = None
        
        if similarity_matrix is not None:
            self.similarity_matrix = similarity_matrix
        elif similarity_matrices_dir is not None and module_name is not None:
            self._load_similarity_matrix(similarity_matrices_dir, module_name, metric)
        else:
            logger.warning("No similarity matrix provided")
    
    def _load_similarity_matrix(self, matrices_dir: str, module_name: str, metric: str = 'cosine'):
        similarity_path = Path(matrices_dir) / 'similarity_matrices.pkl'

        if not similarity_path.exists():
            logger.error(f"Similarity matrix file does not exist: {similarity_path}")
            return

        with open(similarity_path, 'rb') as f:
            similarity_data = pickle.load(f)

        if module_name not in similarity_data:
            logger.error(f"Module {module_name} does not exist in similarity data")
            return

        module_matrices = similarity_data[module_name]
        if metric not in module_matrices:
            logger.error(f"Metric {metric} does not exist in module {module_name}")
            return

        self.similarity_matrix = module_matrices[metric]
        logger.info(f"Successfully loaded {metric} similarity matrix for module {module_name}")
    
    def set_similarity_matrix(self, similarity_matrix: np.ndarray):
        self.similarity_matrix = similarity_matrix
    
    def compute_optimal_steps(self, num_caches: int) -> List[int]:
        if self.similarity_matrix is None:
            logger.error("Similarity matrix not set, cannot compute optimal steps")
            return []

        n_steps = self.similarity_matrix.shape[0]

        # Pre-compute cost matrix: represents the sum of similarities obtained by using cache from timestep a to b
        cost = np.zeros((n_steps, n_steps))
        for a in range(n_steps):
            for b in range(a, n_steps):
                # Compute similarity sum for interval [a,b]
                cost[a, b] = sum(self.similarity_matrix[a, t] for t in range(a, b+1))

        # Determine whether to minimize or maximize
        minimize = self.metric in ['mse', 'l1', '12','wasserstein']

        if minimize:
            # Minimize: initialize to infinity
            dp = np.full((num_caches+1, n_steps), float('inf'))
            logger.info(f"Using {self.metric} metric, will perform minimization")
        else:
            # Maximize: initialize to zero
            dp = np.zeros((num_caches+1, n_steps))
            logger.info(f"Using {self.metric} metric, will perform maximization")

        # Array to record paths
        path = np.zeros((num_caches+1, n_steps), dtype=int)

        # Initialize: case with only one cache point (i.e., initial point)
        for i in range(n_steps):
            dp[1, i] = cost[0, i]

        # Fill DP table
        for k in range(2, num_caches+1):
            for i in range(k-1, n_steps):
                if minimize:
                    best_val = float('inf')
                else:
                    best_val = -float('inf')
                best_j = 0

                # Enumerate end point j of the first k-1 segments
                for j in range(k-2, i):
                    val = dp[k-1, j] + cost[j+1, i]
                    if (minimize and val < best_val) or (not minimize and val > best_val):
                        best_val = val
                        best_j = j

                dp[k, i] = best_val
                path[k, i] = best_j

        # Backtrack to find optimal path
        optimal_steps = []
        k, i = num_caches, n_steps-1

        while k > 0:
            if k == 1:
                # First segment starts directly from 0
                optimal_steps.append(0)
                break
            else:
                j = path[k, i]
                optimal_steps.append(j+1)  # j+1 is the start point of the new segment
                i, k = j, k-1

        # Reverse list to sort by increasing timestep
        optimal_steps = sorted(optimal_steps)

        self.optimal_steps = optimal_steps
        return optimal_steps
    
    def save_optimal_steps(self, save_path: str):

        if self.optimal_steps is None:
            logger.error("Optimal steps not computed, cannot save")
            return

        # Ensure directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        # Save original pkl file
        with open(save_path, 'wb') as f:
            pickle.dump(self.optimal_steps, f)

        # Save as JSON format
        json_path = save_path.replace('.pkl', '.json')

        # Convert NumPy types to Python native types
        # Ensure all values are JSON serializable
        steps_list = [int(step) for step in self.optimal_steps]

        with open(json_path, 'w') as f:
            json.dump({
                'metric': self.metric,
                'steps': steps_list,
                'num_steps': len(steps_list)
            }, f, indent=2)

        # Save as human-readable text format
        txt_path = save_path.replace('.pkl', '.txt')
        with open(txt_path, 'w') as f:
            f.write(f"# Similarity metric: {self.metric}\n")
            f.write(f"# Number of steps: {len(self.optimal_steps)}\n")
            f.write("# Optimal cache steps:\n")

            # Write 10 numbers per line for easy viewing
            for i in range(0, len(self.optimal_steps), 10):
                chunk = self.optimal_steps[i:i+10]
                f.write(" ".join(str(int(step)) for step in chunk) + "\n")

        logger.info(f"Optimal cache steps saved to:\n  - {save_path} (pkl)\n  - {json_path} (json)\n  - {txt_path} (txt)")
    
    @staticmethod
    def load_optimal_steps(load_path: str) -> Optional[List[int]]:
        """Load optimal cache steps from file"""
        if not Path(load_path).exists():
            logger.error(f"Optimal steps file does not exist: {load_path}")
            return None

        with open(load_path, 'rb') as f:
            optimal_steps = pickle.load(f)
        logger.info(f"Loaded optimal cache steps from {load_path}")
        return optimal_steps

def compute_optimal_cache_steps(similarity_matrices_dir: str, module_name: str,
                              num_caches: int, metric: str = 'cosine') -> List[int]:

    scheduler = OptimalCacheScheduler(
        similarity_matrices_dir=similarity_matrices_dir,
        module_name=module_name,
        metric=metric
    )

    optimal_steps = scheduler.compute_optimal_steps(num_caches)

    # Automatically save results to similarity matrices directory
    save_path = Path(similarity_matrices_dir) / 'optimal_steps' / metric / module_name /f"optimal_steps_{module_name}_{num_caches}_{metric}.pkl"
    scheduler.save_optimal_steps(str(save_path))

    return optimal_steps

def main():

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create argument parser
    parser = argparse.ArgumentParser(description='Compute optimal cache steps')

    parser.add_argument('--similarity_matrices_dir', type=str, required=True,
                      help='Path to similarity matrices directory, containing similarity_matrices.pkl file generated by analyze_similarity_matrix.py')

    parser.add_argument('--module_name', type=str, required=True,
                      help='Module name to use, e.g., encoder.layers.0')

    parser.add_argument('--num_caches', type=int, default=30,
                      help='Number of allowed cache updates, default 30')

    parser.add_argument('--metric', type=str, default='cosine',
                      choices=['mse', 'cosine', 'l1', 'l2','wasserstein'],
                      help='Similarity metric to use, default cosine')

    parser.add_argument('--list_modules', action='store_true',
                      help='List all available modules in the similarity matrices')

    # Parse command line arguments
    args = parser.parse_args()

    # If listing module names is requested
    if args.list_modules:
        similarity_path = Path(args.similarity_matrices_dir) / 'similarity_matrices.pkl'
        if not similarity_path.exists():
            logger.error(f"Similarity matrix file does not exist: {similarity_path}")
            sys.exit(1)

        with open(similarity_path, 'rb') as f:
            similarity_data = pickle.load(f)

        print("Available modules:")
        for module_name in similarity_data.keys():
            print(f"  - {module_name}")
            # Display available metrics for this module
            if isinstance(similarity_data[module_name], dict):
                print("    Available metrics:")
                for metric in similarity_data[module_name].keys():
                    print(f"      - {metric}")
        return

    # Compute optimal steps
    logger.info(f"Starting to compute optimal cache steps using module {args.module_name}, allowing {args.num_caches} cache updates...")

    # Check if similarity matrix file exists
    similarity_path = Path(args.similarity_matrices_dir) / 'similarity_matrices.pkl'
    if not similarity_path.exists():
        logger.error(f"Similarity matrix file does not exist: {similarity_path}")
        sys.exit(1)

    # Compute optimal steps
    optimal_steps = compute_optimal_cache_steps(
        similarity_matrices_dir=args.similarity_matrices_dir,
        module_name=args.module_name,
        num_caches=args.num_caches,
        metric=args.metric
    )

    logger.info(f"Optimal cache steps computation complete: {optimal_steps}")

if __name__ == "__main__":
    main() 
