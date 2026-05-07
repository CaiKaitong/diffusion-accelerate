#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

cd "$PROJECT_ROOT"
echo "Working directory: $(pwd)"
DEVICE="cuda:0"
NUM_CACHES="10"
METRIC="cosine"
NUM_BU_BLOCKS="5"
DEBUG_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --num_caches)
            NUM_CACHES="$2"
            shift 2
            ;;
        --metric)
            METRIC="$2"
            shift 2
            ;;
        --task)
            SINGLE_TASK="$2"
            shift 2
            ;;
        --num_bu_blocks)
            NUM_BU_BLOCKS="$2"
            shift 2
            ;;
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done
# declare -A PH_TASKS=(
#     ["lift_ph"]="checkpoint/lift_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=0250-test_mean_score=1.000.ckpt"
#     ["can_ph"]="checkpoint/can_ph/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
#     ["square_ph"]="checkpoint/square_ph/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
#     ["transport_ph"]="checkpoint/transport_ph/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
#     ["tool_hang_ph"]="checkpoint/tool_hang_ph/diffusion_policy_transformer/train_1/checkpoints/epoch=2400-test_mean_score=0.682.ckpt"
#     ["pusht"]="checkpoint/pusht/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
# )

declare -A MH_TASKS=(
    ["lift_mh"]="checkpoint/lift_mh/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["can_mh"]="checkpoint/can_mh/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["square_mh"]="checkpoint/square_mh/diffusion_policy_transformer/train_1/checkpoints/epoch=2950-test_mean_score=0.864.ckpt"
    ["transport_mh"]="checkpoint/transport_mh/diffusion_policy_transformer/train_0/checkpoints/epoch=1750-test_mean_score=0.727.ckpt"
)
# declare -A LOWDIM_TASKS=(
#     ["block_pushing"]="checkpoint/low_dim/block_pushing/diffusion_policy_transformer/train_0/checkpoints/epoch=7550-test_mean_score=1.000.ckpt"
#     ["kitchen"]="checkpoint/low_dim/kitchen/diffusion_policy_transformer/train_0/checkpoints/epoch=3000-test_mean_score=0.574.ckpt"
# )

if [ ! -z "$SINGLE_TASK" ]; then
    if [[ -v PH_TASKS["$SINGLE_TASK"] ]]; then
        echo "Processing PH task: $SINGLE_TASK"
        declare -A PH_TASKS_FILTERED
        PH_TASKS_FILTERED["$SINGLE_TASK"]="${PH_TASKS[$SINGLE_TASK]}"
        PH_TASKS=()
        for key in "${!PH_TASKS_FILTERED[@]}"; do
            PH_TASKS["$key"]="${PH_TASKS_FILTERED[$key]}"
        done
        MH_TASKS=()
        LOWDIM_TASKS=()
    elif [[ -v MH_TASKS["$SINGLE_TASK"] ]]; then
        echo "Processing MH task: $SINGLE_TASK"
        declare -A MH_TASKS_FILTERED
        MH_TASKS_FILTERED["$SINGLE_TASK"]="${MH_TASKS[$SINGLE_TASK]}"
        MH_TASKS=()
        for key in "${!MH_TASKS_FILTERED[@]}"; do
            MH_TASKS["$key"]="${MH_TASKS_FILTERED[$key]}"
        done
        PH_TASKS=()
        LOWDIM_TASKS=()
    elif [[ -v LOWDIM_TASKS["$SINGLE_TASK"] ]]; then
        echo "Processing lowdim task: $SINGLE_TASK"
        declare -A LOWDIM_TASKS_FILTERED
        LOWDIM_TASKS_FILTERED["$SINGLE_TASK"]="${LOWDIM_TASKS[$SINGLE_TASK]}"
        LOWDIM_TASKS=()
        for key in "${!LOWDIM_TASKS_FILTERED[@]}"; do
            LOWDIM_TASKS["$key"]="${LOWDIM_TASKS_FILTERED[$key]}"
        done
        PH_TASKS=()
        MH_TASKS=()
    else
        echo "Error: Task not found $SINGLE_TASK"
        exit 1
    fi
fi

echo "=========================================="
echo "Displaying steps info"
echo "=========================================="
echo "Device: $DEVICE"
echo "Num caches: $NUM_CACHES"
echo "Metric: $METRIC"
echo "BU blocks: $NUM_BU_BLOCKS"
if [ "$DEBUG_MODE" = true ]; then
    echo "Debug mode: ON"
fi
echo "=========================================="

process_task() {
    local task_name=$1
    local task_type=$2
    
    echo "=========================================="
    echo "Processing task: $task_name (type: $task_type)"
    echo "=========================================="
    
    local steps_dir="assets/${task_name}/original/optimal_steps/${METRIC}"
    if [ ! -d "$steps_dir" ]; then
        echo "Error: Steps directory not found: $steps_dir"
        return 1
    fi
    
    local bu_blocks_file="assets/${task_name}/original/bu_block_selection/top_${NUM_BU_BLOCKS}_error_blocks.pkl"
    if [ ! -f "$bu_blocks_file" ]; then
        echo "Error: BU blocks file not found: $bu_blocks_file"
        return 1
    fi
    
    if [ "$DEBUG_MODE" = true ]; then
        echo "Debug mode: Skipping"
        return 0
    fi
    
    python3 -c "
import os
import pickle
import re
from pathlib import Path

def get_layer_idx(block_key):
    match = re.match(r'decoder\.layers\.(\d+)_([a-z_]+)', block_key)
    if match:
        return int(match.group(1))
    match = re.match(r'decoder\.layers\.(\d+)\.dropout\d+', block_key)
    if match:
        return int(match.group(1))
    return -1

def get_block_type(block_key):
    match = re.match(r'decoder\.layers\.\d+_([a-z_]+)', block_key)
    if match:
        return match.group(1)
    match = re.match(r'decoder\.layers\.\d+\.dropout(\d+)', block_key)
    if match:
        dropout_num = int(match.group(1))
        if dropout_num == 1:
            return 'sa_block'
        elif dropout_num == 2:
            return 'mha_block'
        elif dropout_num == 3:
            return 'ff_block'
    return ''

def convert_to_block_format(block_key):
    match = re.match(r'decoder\.layers\.(\d+)\.dropout(\d+)', block_key)
    if match:
        layer_idx = match.group(1)
        dropout_num = int(match.group(2))
        if dropout_num == 1:
            return f'decoder.layers.{layer_idx}_sa_block'
        elif dropout_num == 2:
            return f'decoder.layers.{layer_idx}_mha_block'
        elif dropout_num == 3:
            return f'decoder.layers.{layer_idx}_ff_block'
    return block_key

steps_dir = Path('$steps_dir')
original_steps = {}
for block_dir in steps_dir.iterdir():
    if block_dir.is_dir():
        block_name = block_dir.name
        steps_file = block_dir / f'optimal_steps_{block_name}_{$NUM_CACHES}_${METRIC}.pkl'
        if steps_file.exists():
            with open(steps_file, 'rb') as f:
                steps = pickle.load(f)
                original_steps[block_name] = steps

with open('$bu_blocks_file', 'rb') as f:
    bu_blocks = pickle.load(f)
    bu_blocks = [convert_to_block_format(block) for block in bu_blocks.keys()]

print('\nBU Blocks:')
print('-'*30)
for block in bu_blocks:
    print(f'- {block}')

practical_steps = original_steps.copy()
ffn_blocks = [block for block in original_steps.keys() if 'ff_block' in block]

step_changes = {}

for bu_block in bu_blocks:
    bu_layer = get_layer_idx(bu_block)
    deeper_ffn = [block for block in ffn_blocks if get_layer_idx(block) > bu_layer]
    
    all_deeper_steps = set()
    for deeper_block in deeper_ffn:
        if deeper_block in original_steps:
            all_deeper_steps.update(original_steps[deeper_block])
    
    if bu_block in practical_steps:
        current_steps = set(practical_steps[bu_block])
        new_steps = sorted(list(current_steps.union(all_deeper_steps)))
        practical_steps[bu_block] = new_steps
        
        added_steps = set(new_steps) - current_steps
        if added_steps:
            step_changes[bu_block] = {
                'original': sorted(list(current_steps)),
                'added': sorted(list(added_steps)),
                'final': new_steps
            }

print('\nOriginal Steps (BAC):')
print('-'*30)
for block, steps in sorted(original_steps.items(), key=lambda x: (get_layer_idx(x[0]), get_block_type(x[0]))):
    print(f'{block}: {steps}')

print('\nPractical Steps (with BU=$NUM_BU_BLOCKS):')
print('-'*30)
for block, steps in sorted(practical_steps.items(), key=lambda x: (get_layer_idx(x[0]), get_block_type(x[0]))):
    if block in step_changes:
        print(f'{block}:')
        print(f'  Original: {step_changes[block][\"original\"]}')
        print(f'  Added:    {step_changes[block][\"added\"]}')
        print(f'  Final:    {step_changes[block][\"final\"]}')
    else:
        print(f'{block}: {steps}')
"
}

if [ ${#PH_TASKS[@]} -gt 0 ]; then
    echo "=========================================="
    echo "Processing PH tasks..."
    echo "=========================================="
    
    for task_name in "${!PH_TASKS[@]}"; do
        process_task "$task_name" "PH"
    done
fi

if [ ${#MH_TASKS[@]} -gt 0 ]; then
    echo "=========================================="
    echo "Processing MH tasks..."
    echo "=========================================="
    
    for task_name in "${!MH_TASKS[@]}"; do
        process_task "$task_name" "MH"
    done
fi

if [ ${#LOWDIM_TASKS[@]} -gt 0 ]; then
    echo "=========================================="
    echo "Processing lowdim tasks..."
    echo "=========================================="
    
    for task_name in "${!LOWDIM_TASKS[@]}"; do
        process_task "$task_name" "LOWDIM"
    done
fi

echo "=========================================="
echo "All tasks completed"
echo "=========================================="

usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --device device_name        Device (default: cuda:0)"
    echo "  --num_caches num            Number of caches (default: 10)"
    echo "  --metric metric             Similarity metric (default: cosine)"
    echo "  --num_bu_blocks num         Number of BU blocks (default: 5)"
    echo "  --task task_name            Process specific task only"
    echo "  --debug                     Debug mode"
    echo
    echo "Examples:"
    echo "  $0 --device cuda:1 --num_caches 10 --metric cosine --task block_pushing"
    echo "  $0 --num_bu_blocks 3 --task tool_hang_ph"
    echo "  $0 --task kitchen --debug"
}

if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    usage
    exit 0
fi