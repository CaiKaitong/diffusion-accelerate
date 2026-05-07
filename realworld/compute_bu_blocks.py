#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))
from BACInfer.analysis.bu_block_selection import analyze_block_errors

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)



def find_activations_file(collected_dir: str) -> Path:
    collected_path = Path(collected_dir)
    
    activation_files = list(collected_path.glob("activations_*.pkl"))
    
    if not activation_files:
        raise FileNotFoundError(f"No activations file found in {collected_dir}")
    
    if len(activation_files) > 1:
        activation_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    return activation_files[0]


def compute_and_save_bu_blocks(activations_path: str, num_blocks: int = 3, 
                                output_dir: str = None) -> Path:
    if output_dir is None:
        output_dir = Path(activations_path).parent
    
    bu_dir = Path(output_dir) / "bu_block_selection"
    bu_dir.mkdir(parents=True, exist_ok=True)
    selected_blocks = analyze_block_errors(
        activations_path=str(activations_path),
        output_dir=str(bu_dir),
        num_blocks=num_blocks
    )
    return bu_dir


def main():
    parser = argparse.ArgumentParser(
        description='Compute BU (Bubbling Union) Blocks',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--activations', type=str,
                            help='Path to activations file')
    input_group.add_argument('--collected_dir', type=str,
                            help='Collection directory (auto-find activations file)')
    
    parser.add_argument('--num_blocks', type=int, default=3,
                       help='Number of BU blocks to select (default: 3)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: activations file directory)')
    
    args = parser.parse_args()
    
    if args.activations:
        activations_path = args.activations
        if not Path(activations_path).exists():
            sys.exit(1)
    else:
        try:
            activations_path = find_activations_file(args.collected_dir)
        except FileNotFoundError:
            sys.exit(1)
    
    bu_dir = compute_and_save_bu_blocks(
        activations_path=str(activations_path),
        num_blocks=args.num_blocks,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
