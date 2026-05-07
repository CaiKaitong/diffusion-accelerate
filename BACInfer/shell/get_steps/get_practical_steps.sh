#!/bin/bash

# BUblocksteps
# BU，blocksteps

# 
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# （shs/get_steps/）
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# 
cd "$PROJECT_ROOT"
echo ": $(pwd)"

# 
DEVICE="cuda:0"
NUM_CACHES="7"
METRIC="cosine"
DEBUG_MODE=false

# 
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
            echo ": $1"
            exit 1
            ;;
    esac
done

# BU
if [ -z "$NUM_BU_BLOCKS" ]; then
    NUM_BU_BLOCKS=5
fi

# PH
declare -A PH_TASKS=(
    ["lift_ph"]="checkpoint/lift_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=0250-test_mean_score=1.000.ckpt"
    ["can_ph"]="checkpoint/can_ph/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["square_ph"]="checkpoint/square_ph/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["transport_ph"]="checkpoint/transport_ph/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["tool_hang_ph"]="checkpoint/tool_hang_ph/diffusion_policy_transformer/train_1/checkpoints/epoch=2400-test_mean_score=0.682.ckpt"
    ["pusht"]="checkpoint/pusht/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
)

# MH
declare -A MH_TASKS=(
    ["lift_mh"]="checkpoint/lift_mh/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["can_mh"]="checkpoint/can_mh/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["square_mh"]="checkpoint/square_mh/diffusion_policy_transformer/train_1/checkpoints/epoch=2950-test_mean_score=0.864.ckpt"
    ["transport_mh"]="checkpoint/transport_mh/diffusion_policy_transformer/train_0/checkpoints/epoch=1750-test_mean_score=0.727.ckpt"
)

# 
declare -A LOWDIM_TASKS=(
    ["block_pushing"]="checkpoint/low_dim/block_pushing/diffusion_policy_transformer/train_0/checkpoints/epoch=7550-test_mean_score=1.000.ckpt"
    ["kitchen"]="checkpoint/low_dim/kitchen/diffusion_policy_transformer/train_0/checkpoints/epoch=3000-test_mean_score=0.574.ckpt"
)

# FLOPS
declare -a PH_ORIGINAL_FLOPS=()
declare -a PH_PRACTICAL_FLOPS=()
declare -a MH_ORIGINAL_FLOPS=()
declare -a MH_PRACTICAL_FLOPS=()
declare -a LOWDIM_ORIGINAL_FLOPS=()
declare -a LOWDIM_PRACTICAL_FLOPS=()

# 
if [ ! -z "$SINGLE_TASK" ]; then
    if [[ -v PH_TASKS["$SINGLE_TASK"] ]]; then
        echo "PH: $SINGLE_TASK"
        declare -A PH_TASKS_FILTERED
        PH_TASKS_FILTERED["$SINGLE_TASK"]="${PH_TASKS[$SINGLE_TASK]}"
        PH_TASKS=()
        for key in "${!PH_TASKS_FILTERED[@]}"; do
            PH_TASKS["$key"]="${PH_TASKS_FILTERED[$key]}"
        done
        MH_TASKS=()
        LOWDIM_TASKS=()
    elif [[ -v MH_TASKS["$SINGLE_TASK"] ]]; then
        echo "MH: $SINGLE_TASK"
        declare -A MH_TASKS_FILTERED
        MH_TASKS_FILTERED["$SINGLE_TASK"]="${MH_TASKS[$SINGLE_TASK]}"
        MH_TASKS=()
        for key in "${!MH_TASKS_FILTERED[@]}"; do
            MH_TASKS["$key"]="${MH_TASKS_FILTERED[$key]}"
        done
        PH_TASKS=()
        LOWDIM_TASKS=()
    elif [[ -v LOWDIM_TASKS["$SINGLE_TASK"] ]]; then
        echo ": $SINGLE_TASK"
        declare -A LOWDIM_TASKS_FILTERED
        LOWDIM_TASKS_FILTERED["$SINGLE_TASK"]="${LOWDIM_TASKS[$SINGLE_TASK]}"
        LOWDIM_TASKS=()
        for key in "${!LOWDIM_TASKS_FILTERED[@]}"; do
            LOWDIM_TASKS["$key"]="${LOWDIM_TASKS_FILTERED[$key]}"
        done
        PH_TASKS=()
        MH_TASKS=()
    else
        echo ":  $SINGLE_TASK"
        exit 1
    fi
fi

# 
echo "===========================================" 
echo "BUblocksteps"
echo "===========================================" 
echo ": $DEVICE"
echo ": $NUM_CACHES"
echo ": $METRIC"
echo "BU: $NUM_BU_BLOCKS"
if [ "$DEBUG_MODE" = true ]; then
    echo ":  (，)"
fi
echo "===========================================" 

# 
RESULT_DIR="results/practical_steps"
mkdir -p "$RESULT_DIR"

# : steps
process_task() {
    local task_name=$1
    local checkpoint=$2
    local task_type=$3  # ： (PH/MH/LOWDIM)
    
    echo ": $task_name (: $task_type)"
    echo ": $checkpoint"
    
    # 
    local steps_dir="assets/${task_name}/original/optimal_steps"
    if [ ! -d "$steps_dir" ]; then
        echo ": : $steps_dir"
        return 1
    fi
    
    # 
    local metric_dir="${steps_dir}/${METRIC}"
    if [ ! -d "$metric_dir" ]; then
        echo ": : $metric_dir"
        echo ": $(ls $steps_dir)"
        return 1
    fi
    
    # BU
    local bu_blocks_file="assets/${task_name}/original/bu_block_selection/top_${NUM_BU_BLOCKS}_error_blocks.pkl"
    if [ ! -f "$bu_blocks_file" ]; then
        echo ": BU: $bu_blocks_file"
        echo "BU: $(ls assets/${task_name}/original/bu_block_selection/)"
        return 1
    fi
    
    # ，
    if [ "$DEBUG_MODE" = true ]; then
        echo ": "
        return 0
    fi
    
    # 
    local output_file="${RESULT_DIR}/${task_name}_bu${NUM_BU_BLOCKS}_${METRIC}_caches${NUM_CACHES}_practical_steps.csv"
    
    # Python
    # ：bashPython
    output=$(python -c "
import os
import re
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# bash
NUM_CACHES = $NUM_CACHES
METRIC = \"$METRIC\"
TASK_TYPE = \"$task_type\"  # 

# ，diffusion_cache_wrapper.py
def get_layer_idx(block_key):
    match = re.match(r'decoder\.layers\.(\d+)_([a-z_]+)', block_key)
    if match:
        return int(match.group(1))
    
    #  decoder.layers.X.dropoutY 
    match = re.match(r'decoder\.layers\.(\d+)\.dropout\d+', block_key)
    if match:
        return int(match.group(1))
    
    return -1

def get_block_type_priority(block_key):
    match = re.match(r'decoder\.layers\.\d+_([a-z_]+)', block_key)
    if match:
        block_type = match.group(1)
        priorities = {'sa_block': 0, 'mha_block': 1, 'ff_block': 2}
        return priorities.get(block_type, 10)  # 
    
    #  decoder.layers.X.dropoutY 
    match = re.match(r'decoder\.layers\.\d+\.dropout(\d+)', block_key)
    if match:
        dropout_num = int(match.group(1))
        if dropout_num == 1:
            return 0  # sa_block
        elif dropout_num == 2:
            return 1  # mha_block
        elif dropout_num == 3:
            return 2  # ff_block
    
    return 10

def get_block_type(block_key):
    match = re.match(r'decoder\.layers\.\d+_([a-z_]+)', block_key)
    if match:
        return match.group(1)
    
    #  decoder.layers.X.dropoutY 
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

# FLOPS
def calculate_flops(ff_steps, sa_steps, mha_steps):
    
    base_flops = 0.4116
    ff_coeff = 0.01048
    sa_coeff = 0.005294
    mha_coeff = 0.003424
    
    return base_flops + (ff_coeff * ff_steps) + (sa_coeff * sa_steps) + (mha_coeff * mha_steps)

#  ( dropout  block )
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
    
    return block_key  # ，

# BU
bu_blocks_file = \"$bu_blocks_file\"
bu_blocks = []
try:
    with open(bu_blocks_file, 'rb') as f:
        bu_blocks_dict = pickle.load(f)
        bu_blocks = list(bu_blocks_dict.keys())
        # BU
        bu_blocks = [convert_to_block_format(block) for block in bu_blocks]
    print(f' {len(bu_blocks)} BU: {bu_blocks}')
except Exception as e:
    print(f'BU: {e}')
    bu_blocks = []

# 
steps_dir = Path(\"$metric_dir\")  # 
practical_steps = {}
block_info = {}

# : 
print(f': {list(steps_dir.iterdir()) if steps_dir.exists() else ""}')

# block，
if not steps_dir.exists():
    print(f': : {steps_dir}')
    exit(1)
    
for block_dir in steps_dir.iterdir():
    if block_dir.is_dir():
        original_block_key = block_dir.name
        block_key = convert_to_block_format(original_block_key)
        block_type = get_block_type(block_key)
        layer_idx = get_layer_idx(block_key)
        
        print(f': {original_block_key} -> : {block_key}, : {block_type}, : {layer_idx}')
        
        # 
        steps_file = block_dir / f'optimal_steps_{original_block_key}_{NUM_CACHES}_{METRIC}.pkl'
        if steps_file.exists():
            try:
                with open(steps_file, 'rb') as f:
                    steps = pickle.load(f)
                if block_key not in practical_steps:
                    practical_steps[block_key] = steps
                    block_info[block_key] = {
                        'block_type': block_type,
                        'layer_idx': layer_idx,
                        'original_steps': steps,
                        'is_bu_block': block_key in bu_blocks,
                        'practical_steps': steps  # steps
                    }
                    print(f' {block_key} : {steps}')
            except Exception as e:
                print(f' {steps_file}: {e}')

# BU，
if not practical_steps:
    print('，')
    exit(1)
    
if len(bu_blocks) == 0:
    print('BU，BU')
    exit(1)

# 
print(f' {len(practical_steps)} :')
for block_key in practical_steps.keys():
    print(f'  - {block_key} (: {get_block_type(block_key)})')

# BU，diffusion_cache_wrapper.py
if len(bu_blocks) > 1:
    # BU
    sorted_bu_blocks = sorted(bu_blocks, key=lambda x: (get_layer_idx(x), get_block_type_priority(x)))
    print(f'BU: : {sorted_bu_blocks}')
    
    # FFN（bu_blocks）
    all_ffn_blocks = []
    for block_key in practical_steps.keys():
        if get_block_type(block_key) == 'ff_block':
            all_ffn_blocks.append(block_key)
    
    sorted_all_ffn_blocks = sorted(all_ffn_blocks, key=lambda x: get_layer_idx(x))
    print(f'，BU: FFN（）: {sorted_all_ffn_blocks}')
    
    # bu_blocksFFN Block
    for i, block_key in enumerate(sorted_bu_blocks):
        block_layer_idx = get_layer_idx(block_key)
        print(f'BU: {block_key}, : {block_layer_idx}')
        
        # FFN Block
        deeper_ffn_blocks = []
        for ffn_block in sorted_all_ffn_blocks:
            ffn_layer_idx = get_layer_idx(ffn_block)
            if ffn_layer_idx >= block_layer_idx:
                deeper_ffn_blocks.append(ffn_block)
        
        if deeper_ffn_blocks:
            print(f'BU:  {block_key} ( {block_layer_idx}) FFN: {deeper_ffn_blocks}')
            
            # FFN Blocksteps
            all_deeper_steps = set()
            for deeper_ffn_block in deeper_ffn_blocks:
                if deeper_ffn_block in practical_steps:
                    all_deeper_steps.update(practical_steps[deeper_ffn_block])
            
            # FFN Blocksteps
            if block_key in block_info and all_deeper_steps:
                current_steps = set(block_info[block_key]['original_steps'])
                missing_steps = all_deeper_steps - current_steps
                
                if missing_steps:
                    updated_steps = sorted(list(current_steps.union(missing_steps)))
                    block_info[block_key]['practical_steps'] = updated_steps
                    practical_steps[block_key] = updated_steps
                    print(f'BU:  {block_key} FFN Blocksteps: {sorted(list(missing_steps))}。steps: {updated_steps}')
        else:
            print(f' {block_key} FFN')
else:
    print('BU: （2）。')

# 
results = []
for block_key, info in block_info.items():
    original_count = len(info['original_steps'])
    practical_count = len(info['practical_steps'])
    
    results.append({
        'block_key': block_key,
        'block_type': info['block_type'],
        'layer_idx': info['layer_idx'],
        'is_bu_block': info['is_bu_block'],
        'original_steps': original_count,
        'practical_steps': practical_count,
        'added_steps': practical_count - original_count,
        'increase_percent': (practical_count - original_count) / original_count * 100 if original_count > 0 else 0
    })

# 
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by=['block_type', 'layer_idx'])

# 
summary = results_df.groupby('block_type').agg({
    'original_steps': 'sum',
    'practical_steps': 'sum',
    'added_steps': 'sum',
    'increase_percent': 'mean'
}).reset_index()

# 
total_row = {
    'block_type': 'total',
    'original_steps': results_df['original_steps'].sum(),
    'practical_steps': results_df['practical_steps'].sum(),
    'added_steps': results_df['added_steps'].sum(),
    'increase_percent': results_df['increase_percent'].mean()
}
summary = pd.concat([summary, pd.DataFrame([total_row])], ignore_index=True)

# FLOPS
# 
ff_original_steps = summary.loc[summary['block_type'] == 'ff_block', 'original_steps'].values[0] if 'ff_block' in summary['block_type'].values else 0
sa_original_steps = summary.loc[summary['block_type'] == 'sa_block', 'original_steps'].values[0] if 'sa_block' in summary['block_type'].values else 0
mha_original_steps = summary.loc[summary['block_type'] == 'mha_block', 'original_steps'].values[0] if 'mha_block' in summary['block_type'].values else 0

ff_practical_steps = summary.loc[summary['block_type'] == 'ff_block', 'practical_steps'].values[0] if 'ff_block' in summary['block_type'].values else 0
sa_practical_steps = summary.loc[summary['block_type'] == 'sa_block', 'practical_steps'].values[0] if 'sa_block' in summary['block_type'].values else 0
mha_practical_steps = summary.loc[summary['block_type'] == 'mha_block', 'practical_steps'].values[0] if 'mha_block' in summary['block_type'].values else 0

# FLOPS
original_flops = calculate_flops(ff_original_steps, sa_original_steps, mha_original_steps)
practical_flops = calculate_flops(ff_practical_steps, sa_practical_steps, mha_practical_steps)

# FLOPS
flops_increase = practical_flops - original_flops
flops_increase_percent = (flops_increase / original_flops) * 100 if original_flops > 0 else 0

# FLOPS
# f-string
task_name = \"$task_name\"
print(\"\\n===== \" + TASK_TYPE + \": \" + task_name + \" FLOPS =====\")
print(f'FLOPS: {original_flops:.6f}')
print(f'FLOPS: {practical_flops:.6f}')
print(f'FLOPS: {flops_increase:.6f} ({flops_increase_percent:.2f}%)')
print('===========================\\n')

# FLOPS
flops_row = {
    'block_type': 'flops',
    'original_steps': original_flops,
    'practical_steps': practical_flops,
    'added_steps': flops_increase,
    'increase_percent': flops_increase_percent
}
summary = pd.concat([summary, pd.DataFrame([flops_row])], ignore_index=True)

# 
output_file = \"$output_file\"
results_df.to_csv(output_file, index=False)
print(f': {output_file}')

# 
summary_file = output_file.replace('.csv', '_summary.csv')
summary.to_csv(summary_file, index=False)
print(f': {summary_file}')

# 
print('\\n=====  =====')
print(summary)
print('===================\\n')

# FLOPSbash
print(f'FLOPS_RESULTS:{original_flops}:{practical_flops}')
" 2>&1) || {
        echo "Python"
        return 1
    }
    
    echo "$output"
    
    # FLOPS，"FLOPS_RESULTS:FLOPS:FLOPS"
    flops_line=$(echo "$output" | grep "FLOPS_RESULTS:" | tail -1)
    if [[ ! -z "$flops_line" ]]; then
        original_flops=$(echo "$flops_line" | cut -d':' -f2)
        practical_flops=$(echo "$flops_line" | cut -d':' -f3)
        
        # FLOPS
        if [[ "$task_type" == "PH" ]]; then
            PH_ORIGINAL_FLOPS+=("$original_flops")
            PH_PRACTICAL_FLOPS+=("$practical_flops")
        elif [[ "$task_type" == "MH" ]]; then
            MH_ORIGINAL_FLOPS+=("$original_flops")
            MH_PRACTICAL_FLOPS+=("$practical_flops")
        elif [[ "$task_type" == "LOWDIM" ]]; then
            LOWDIM_ORIGINAL_FLOPS+=("$original_flops")
            LOWDIM_PRACTICAL_FLOPS+=("$practical_flops")
        fi
    fi
    
    echo " ${task_name} "
    echo "-------------------------------------------"
}

# 
calculate_average() {
    local sum=0
    local count=0
    
    for value in "$@"; do
        sum=$(echo "$sum + $value" | bc -l)
        count=$((count + 1))
    done
    
    if [[ $count -eq 0 ]]; then
        echo "0"
    else
        echo "scale=6; $sum / $count" | bc -l
    fi
}

# PH
if [ ${#PH_TASKS[@]} -gt 0 ]; then
    echo "===========================================" 
    echo "PH..."
    echo "===========================================" 
    
    for task_name in "${!PH_TASKS[@]}"; do
        checkpoint="${PH_TASKS[$task_name]}"
        process_task "$task_name" "$checkpoint" "PH"
    done
fi

# MH
if [ ${#MH_TASKS[@]} -gt 0 ]; then
    echo "===========================================" 
    echo "MH..."
    echo "===========================================" 
    
    for task_name in "${!MH_TASKS[@]}"; do
        checkpoint="${MH_TASKS[$task_name]}"
        process_task "$task_name" "$checkpoint" "MH"
    done
fi

# 
if [ ${#LOWDIM_TASKS[@]} -gt 0 ]; then
    echo "===========================================" 
    echo "..."
    echo "===========================================" 
    
    for task_name in "${!LOWDIM_TASKS[@]}"; do
        checkpoint="${LOWDIM_TASKS[$task_name]}"
        process_task "$task_name" "$checkpoint" "LOWDIM"
    done
fi

# FLOPS
ph_avg_original_flops=0
ph_avg_practical_flops=0
mh_avg_original_flops=0
mh_avg_practical_flops=0
lowdim_avg_original_flops=0
lowdim_avg_practical_flops=0

# bc
if [ ${#PH_ORIGINAL_FLOPS[@]} -gt 0 ]; then
    ph_avg_original_flops=$(calculate_average "${PH_ORIGINAL_FLOPS[@]}")
    ph_avg_practical_flops=$(calculate_average "${PH_PRACTICAL_FLOPS[@]}")
fi

if [ ${#MH_ORIGINAL_FLOPS[@]} -gt 0 ]; then
    mh_avg_original_flops=$(calculate_average "${MH_ORIGINAL_FLOPS[@]}")
    mh_avg_practical_flops=$(calculate_average "${MH_PRACTICAL_FLOPS[@]}")
fi

if [ ${#LOWDIM_ORIGINAL_FLOPS[@]} -gt 0 ]; then
    lowdim_avg_original_flops=$(calculate_average "${LOWDIM_ORIGINAL_FLOPS[@]}")
    lowdim_avg_practical_flops=$(calculate_average "${LOWDIM_PRACTICAL_FLOPS[@]}")
fi

# FLOPS
all_original_flops=("${PH_ORIGINAL_FLOPS[@]}" "${MH_ORIGINAL_FLOPS[@]}" "${LOWDIM_ORIGINAL_FLOPS[@]}")
all_practical_flops=("${PH_PRACTICAL_FLOPS[@]}" "${MH_PRACTICAL_FLOPS[@]}" "${LOWDIM_PRACTICAL_FLOPS[@]}")

all_avg_original_flops=0
all_avg_practical_flops=0

if [ ${#all_original_flops[@]} -gt 0 ]; then
    all_avg_original_flops=$(calculate_average "${all_original_flops[@]}")
    all_avg_practical_flops=$(calculate_average "${all_practical_flops[@]}")
fi

# （）
safe_percentage() {
    local original=$1
    local practical=$2
    
    if (( $(echo "$original > 0" | bc -l) )); then
        echo "scale=2; ($practical - $original) / $original * 100" | bc -l
    else
        echo "0"
    fi
}

# FLOPS
echo "===========================================" 
echo "FLOPS"
echo "===========================================" 
echo "PH(${#PH_ORIGINAL_FLOPS[@]}):"
echo "  FLOPS: $ph_avg_original_flops"
echo "  FLOPS: $ph_avg_practical_flops"
echo "  FLOPS: $(safe_percentage "$ph_avg_original_flops" "$ph_avg_practical_flops")%"
echo

echo "MH(${#MH_ORIGINAL_FLOPS[@]}):"
echo "  FLOPS: $mh_avg_original_flops"
echo "  FLOPS: $mh_avg_practical_flops"
echo "  FLOPS: $(safe_percentage "$mh_avg_original_flops" "$mh_avg_practical_flops")%"
echo

echo "(${#LOWDIM_ORIGINAL_FLOPS[@]}):"
echo "  FLOPS: $lowdim_avg_original_flops"
echo "  FLOPS: $lowdim_avg_practical_flops"
echo "  FLOPS: $(safe_percentage "$lowdim_avg_original_flops" "$lowdim_avg_practical_flops")%"
echo

echo "(${#all_original_flops[@]}):"
echo "  FLOPS: $all_avg_original_flops"
echo "  FLOPS: $all_avg_practical_flops"
echo "  FLOPS: $(safe_percentage "$all_avg_original_flops" "$all_avg_practical_flops")%"
echo "===========================================" 

# FLOPSCSV
avg_flops_file="${RESULT_DIR}/average_flops_summary.csv"
echo "Task_Type,Count,Avg_Original_FLOPS,Avg_Practical_FLOPS,FLOPS_Increase_Percent" > "$avg_flops_file"
echo "PH,${#PH_ORIGINAL_FLOPS[@]},$ph_avg_original_flops,$ph_avg_practical_flops,$(safe_percentage "$ph_avg_original_flops" "$ph_avg_practical_flops")" >> "$avg_flops_file"
echo "MH,${#MH_ORIGINAL_FLOPS[@]},$mh_avg_original_flops,$mh_avg_practical_flops,$(safe_percentage "$mh_avg_original_flops" "$mh_avg_practical_flops")" >> "$avg_flops_file"
echo "LOWDIM,${#LOWDIM_ORIGINAL_FLOPS[@]},$lowdim_avg_original_flops,$lowdim_avg_practical_flops,$(safe_percentage "$lowdim_avg_original_flops" "$lowdim_avg_practical_flops")" >> "$avg_flops_file"
echo "ALL,${#all_original_flops[@]},$all_avg_original_flops,$all_avg_practical_flops,$(safe_percentage "$all_avg_original_flops" "$all_avg_practical_flops")" >> "$avg_flops_file"
echo "FLOPS: $avg_flops_file"

echo ""
echo "===========================================" 

# 
usage() {
    echo ": $0 []"
    echo ":"
    echo "  --device device_name         (: cuda:0)"
    echo "  --num_caches num             (: 10)"
    echo "  --metric metric              (: cosine)"
    echo "  --num_bu_blocks num         BU (: 5)"
    echo "  --task task_name            "
    echo "  --debug                     ，，"
    echo
    echo ":"
    echo "  $0 --device cuda:1 --num_caches 10 --metric cosine --task block_pushing"
    echo "  $0 --num_bu_blocks 3 --task tool_hang_ph"
    echo "  $0 --task kitchen --debug"
}

# 
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    usage
    exit 0
fi
