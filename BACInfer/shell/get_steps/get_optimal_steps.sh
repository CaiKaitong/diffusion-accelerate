#!/bin/bash

# optimal steps
# (PH/MH/)

# 
DEVICE="cuda:0"
NUM_CACHES="10"
METRICS="cosine"
FORCE_RECOMPUTE="--force_recompute"
DEBUG_MODE=false  # 

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
        --metrics)
            METRICS="$2"
            shift 2
            ;;
        --no_force)
            FORCE_RECOMPUTE=""
            shift
            ;;
        --ph_only)
            PH_ONLY=true
            shift
            ;;
        --mh_only)
            MH_ONLY=true
            shift
            ;;
        --lowdim_only)
            LOWDIM_ONLY=true
            shift
            ;;
        --task)
            SINGLE_TASK="$2"
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

# # PH
declare -A PH_TASKS=(
    ["lift_ph"]="checkpoint/lift_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=0250-test_mean_score=1.000.ckpt"
    ["can_ph"]="checkpoint/can_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=0650-test_mean_score=1.000.ckpt"
    ["square_ph"]="checkpoint/square_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=3000-test_mean_score=1.000.ckpt"
    ["transport_ph"]="checkpoint/transport_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=3350-test_mean_score=0.955.ckpt"
    ["tool_hang_ph"]="checkpoint/tool_hang_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=1000-test_mean_score=0.682.ckpt"
    ["pusht"]="checkpoint/pusht/diffusion_policy_transformer/train_2/checkpoints/epoch=0150-test_mean_score=0.752.ckpt"
)

# MH
declare -A MH_TASKS=(
    ["lift_mh"]="checkpoint/lift_mh/diffusion_policy_transformer/train_2/checkpoints/epoch=0250-test_mean_score=1.000.ckpt"
    ["can_mh"]="checkpoint/can_mh/diffusion_policy_transformer/train_2/checkpoints/epoch=0500-test_mean_score=1.000.ckpt"
    ["square_mh"]="checkpoint/square_mh/diffusion_policy_transformer/train_2/checkpoints/epoch=3050-test_mean_score=0.955.ckpt"
    ["transport_mh"]="checkpoint/transport_mh/diffusion_policy_transformer/train_0/checkpoints/epoch=1750-test_mean_score=0.727.ckpt"
)

# 
declare -A LOWDIM_TASKS=(
    # Block Pushing (BP)
    ["block_pushing"]="checkpoint/block_pushing/diffusion_policy_transformer/train_0/checkpoints/epoch=7550-test_mean_score=1.000.ckpt"
    
    # Kitchen
    ["kitchen"]="checkpoint/kitchen/diffusion_policy_transformer/train_0/checkpoints/epoch=3000-test_mean_score=0.574.ckpt"
)



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
    # Block Pushing (BP)
    ["block_pushing"]="checkpoint/low_dim/block_pushing/diffusion_policy_transformer/train_0/checkpoints/epoch=7550-test_mean_score=1.000.ckpt"
    
    # Kitchen
    ["kitchen"]="checkpoint/low_dim/kitchen/diffusion_policy_transformer/train_0/checkpoints/epoch=3000-test_mean_score=0.574.ckpt"
)






# 
if [ ! -z "$SINGLE_TASK" ]; then
    if [[ -v PH_TASKS["$SINGLE_TASK"] ]]; then
        echo "PH: $SINGLE_TASK"
        # ，
        declare -A PH_TASKS_FILTERED
        PH_TASKS_FILTERED["$SINGLE_TASK"]="${PH_TASKS[$SINGLE_TASK]}"
        PH_TASKS=()
        # 
        for key in "${!PH_TASKS_FILTERED[@]}"; do
            PH_TASKS["$key"]="${PH_TASKS_FILTERED[$key]}"
        done
        # 
        MH_TASKS=()
        LOWDIM_TASKS=()
    elif [[ -v MH_TASKS["$SINGLE_TASK"] ]]; then
        echo "MH: $SINGLE_TASK"
        # ，
        declare -A MH_TASKS_FILTERED
        MH_TASKS_FILTERED["$SINGLE_TASK"]="${MH_TASKS[$SINGLE_TASK]}"
        MH_TASKS=()
        # 
        for key in "${!MH_TASKS_FILTERED[@]}"; do
            MH_TASKS["$key"]="${MH_TASKS_FILTERED[$key]}"
        done
        # 
        PH_TASKS=()
        LOWDIM_TASKS=()
    elif [[ -v LOWDIM_TASKS["$SINGLE_TASK"] ]]; then
        echo ": $SINGLE_TASK"
        # ，
        declare -A LOWDIM_TASKS_FILTERED
        LOWDIM_TASKS_FILTERED["$SINGLE_TASK"]="${LOWDIM_TASKS[$SINGLE_TASK]}"
        LOWDIM_TASKS=()
        # 
        for key in "${!LOWDIM_TASKS_FILTERED[@]}"; do
            LOWDIM_TASKS["$key"]="${LOWDIM_TASKS_FILTERED[$key]}"
        done
        # 
        PH_TASKS=()
        MH_TASKS=()
    else
        echo ":  $SINGLE_TASK"
        exit 1
    fi
elif [ "$PH_ONLY" = true ]; then
    echo "PH"
    declare -A MH_TASKS=()
    declare -A LOWDIM_TASKS=()
elif [ "$MH_ONLY" = true ]; then
    echo "MH"
    declare -A PH_TASKS=()
    declare -A LOWDIM_TASKS=()
elif [ "$LOWDIM_ONLY" = true ]; then
    echo ""
    declare -A PH_TASKS=()
    declare -A MH_TASKS=()
fi

# 
echo "===========================================" 
echo ""
echo "===========================================" 
echo ": $DEVICE"
echo ": $NUM_CACHES"
echo ": $METRICS"
if [ "$DEBUG_MODE" = true ]; then
    echo ":  (，)"
fi
if [ ! -z "$FORCE_RECOMPUTE" ]; then
    echo ": "
else
    echo ": "
fi
echo "===========================================" 

# assets
mkdir -p assets

# PH
if [ ${#PH_TASKS[@]} -gt 0 ]; then
    echo "===========================================" 
    echo "PH..."
    echo "===========================================" 
    
    for task_name in "${!PH_TASKS[@]}"; do
        checkpoint="${PH_TASKS[$task_name]}"
        if [ ! -f "$checkpoint" ]; then
            echo ": : $checkpoint, "
            continue
        fi

        output_dir="assets/${task_name}"
        mkdir -p "$output_dir"
        
        echo "PH: $task_name"
        echo ": $checkpoint"
        echo ": $output_dir"
        
        # 
        if [ "$DEBUG_MODE" = true ]; then
            echo ": "
            continue
        fi
        
        # 
        cmd="python -m BACInfer.analysis.get_optimal_cache_update_steps -c $checkpoint -o $output_dir -d $DEVICE --num_caches $NUM_CACHES --metrics \"${METRICS}\" ${FORCE_RECOMPUTE}"
        echo ": $cmd"
        eval $cmd
        
        echo " ${task_name} "
        echo "-------------------------------------------"
    done
fi

# MH
if [ ${#MH_TASKS[@]} -gt 0 ]; then
    echo "===========================================" 
    echo "MH..."
    echo "===========================================" 
    
    for task_name in "${!MH_TASKS[@]}"; do
        checkpoint="${MH_TASKS[$task_name]}"
        if [ ! -f "$checkpoint" ]; then
            echo ": : $checkpoint, "
            continue
        fi
        output_dir="assets/${task_name}"
        mkdir -p "$output_dir"
        
        echo "MH: $task_name"
        echo ": $checkpoint"
        echo ": $output_dir"
        
        # 
        if [ "$DEBUG_MODE" = true ]; then
            echo ": "
            continue
        fi
        
        # 
        cmd="python -m BACInfer.analysis.get_optimal_cache_update_steps -c $checkpoint -o $output_dir -d $DEVICE --num_caches $NUM_CACHES --metrics \"${METRICS}\" ${FORCE_RECOMPUTE}"
        echo ": $cmd"
        eval $cmd
        
        echo " ${task_name} "
        echo "-------------------------------------------"
    done
fi

# 
if [ ${#LOWDIM_TASKS[@]} -gt 0 ]; then
    echo "===========================================" 
    echo "..."
    echo "===========================================" 
    
    for task_name in "${!LOWDIM_TASKS[@]}"; do
        checkpoint="${LOWDIM_TASKS[$task_name]}"
        if [ ! -f "$checkpoint" ]; then
            echo ": : $checkpoint, "
            continue
        fi
        output_dir="assets/${task_name}"
        mkdir -p "$output_dir"
        
        echo ": $task_name"
        echo ": $checkpoint"
        echo ": $output_dir"
        
        # 
        if [ "$DEBUG_MODE" = true ]; then
            echo ": "
            continue
        fi
        
        # 
        cmd="python -m BACInfer.analysis.get_optimal_cache_update_steps -c $checkpoint -o $output_dir -d $DEVICE --num_caches $NUM_CACHES --metrics \"${METRICS}\" ${FORCE_RECOMPUTE}"
        echo ": $cmd"
        eval $cmd
        
        echo " ${task_name} "
        echo "-------------------------------------------"
    done
fi

echo "===========================================" 
echo ""
echo "===========================================" 

# 
usage() {
    echo ": $0 []"
    echo ":"
    echo "  --device device_name         (: cuda:0)"
    echo "  --num_caches list           ， (: 5,8,10,20)"
    echo "  --metrics list              ， (: cosine,l1,mse)"
    echo "  --no_force                  "
    echo "  --ph_only                   PH"
    echo "  --mh_only                   MH"
    echo "  --lowdim_only               (block_pushingkitchen)"
    echo "  --task task_name            "
    echo "  --debug                     ，，"
    echo
    echo ":"
    echo "  $0 --device cuda:1 --num_caches 8,10,15 --metrics cosine --task block_pushing"
    echo "  $0 --device cuda:0 --lowdim_only --num_caches 8,10,15"
    echo "  $0 --lowdim_only --task kitchen --debug"
}

# 
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    usage
    exit 0
fi
