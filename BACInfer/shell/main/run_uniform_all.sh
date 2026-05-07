DEVICE="cuda:0"
TASK_TYPE="all"
CACHE_MODE="threshold"
SKIP_VIDEO="--skip_video"
STEPS=7
OUTPUT_DIR=""
SPECIFIC_TASKS=""
SEEDS="0,1,2"
FORCE_RECOMPUTE=false
declare -a PH_TASKS=(
    "lift_ph"
    "can_ph"
    "square_ph"
    "pusht"
    "transport_ph"
    "tool_hang_ph"
)

declare -a MH_TASKS=(
    "lift_mh"
    "can_mh"
    "square_mh"
    "transport_mh"
)

declare -a LOWDIM_TASKS=(
    "block_pushing"
    "kitchen"
)

declare -a ACTIVE_TASKS=()

declare -A MAX_CHECKPOINTS_SEED0=
    ["lift_ph"]="checkpoint/lift_ph/diffusion_policy_transformer/train_0/checkpoints/epoch=0250-test_mean_score=1.000.ckpt"
    ["can_ph"]="checkpoint/can_ph/diffusion_policy_transformer/train_0/checkpoints/epoch=0950-test_mean_score=1.000.ckpt"
    ["square_ph"]="checkpoint/square_ph/diffusion_policy_transformer/train_0/checkpoints/epoch=2400-test_mean_score=1.000.ckpt"
    ["transport_ph"]="checkpoint/transport_ph/diffusion_policy_transformer/train_0/checkpoints/epoch=2100-test_mean_score=0.955.ckpt"
    ["tool_hang_ph"]="checkpoint/tool_hang_ph/diffusion_policy_transformer/train_0/checkpoints/epoch=0100-test_mean_score=0.773.ckpt"
    ["pusht"]="checkpoint/pusht/diffusion_policy_transformer/train_0/checkpoints/epoch=0100-test_mean_score=0.748.ckpt"
    ["lift_mh"]="checkpoint/lift_mh/diffusion_policy_transformer/train_0/checkpoints/epoch=0250-test_mean_score=1.000.ckpt"
    ["can_mh"]="checkpoint/can_mh/diffusion_policy_transformer/train_0/checkpoints/epoch=1500-test_mean_score=1.000.ckpt"
    ["square_mh"]="checkpoint/square_mh/diffusion_policy_transformer/train_0/checkpoints/epoch=3050-test_mean_score=1.000.ckpt"
    ["transport_mh"]="checkpoint/transport_mh/diffusion_policy_transformer/train_0/checkpoints/epoch=1750-test_mean_score=0.727.ckpt"
    ["block_pushing"]="checkpoint/low_dim/block_pushing/diffusion_policy_transformer/train_0/checkpoints/epoch=7550-test_mean_score=1.000.ckpt"
    ["kitchen"]="checkpoint/low_dim/kitchen/diffusion_policy_transformer/train_0/checkpoints/epoch=3000-test_mean_score=0.574.ckpt"
)

declare -A MAX_CHECKPOINTS_SEED1=(
    ["lift_ph"]="checkpoint/lift_ph/diffusion_policy_transformer/train_1/checkpoints/epoch=0250-test_mean_score=1.000.ckpt"
    ["can_ph"]="checkpoint/can_ph/diffusion_policy_transformer/train_1/checkpoints/epoch=1150-test_mean_score=1.000.ckpt"
    ["square_ph"]="checkpoint/square_ph/diffusion_policy_transformer/train_1/checkpoints/epoch=2200-test_mean_score=1.000.ckpt"
    ["transport_ph"]="checkpoint/transport_ph/diffusion_policy_transformer/train_1/checkpoints/epoch=1800-test_mean_score=0.955.ckpt"
    ["pusht"]="checkpoint/pusht/diffusion_policy_transformer/train_1/checkpoints/epoch=0400-test_mean_score=0.817.ckpt"
    ["lift_mh"]="checkpoint/lift_mh/diffusion_policy_transformer/train_1/checkpoints/epoch=0250-test_mean_score=1.000.ckpt"
    ["can_mh"]="checkpoint/can_mh/diffusion_policy_transformer/train_1/checkpoints/epoch=1100-test_mean_score=1.000.ckpt"
    ["square_mh"]="checkpoint/square_mh/diffusion_policy_transformer/train_1/checkpoints/epoch=2950-test_mean_score=0.864.ckpt"
    ["transport_mh"]="checkpoint/transport_mh/diffusion_policy_transformer/train_1/checkpoints/epoch=0200-test_mean_score=0.773.ckpt"
    ["block_pushing"]="checkpoint/low_dim/block_pushing/diffusion_policy_transformer/train_1/checkpoints/epoch=7950-test_mean_score=1.000.ckpt"
    ["kitchen"]="checkpoint/low_dim/kitchen/diffusion_policy_transformer/train_1/checkpoints/epoch=2700-test_mean_score=0.574.ckpt"
)

declare -A MAX_CHECKPOINTS_SEED2=(
    ["lift_ph"]="checkpoint/lift_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=0250-test_mean_score=1.000.ckpt"
    ["can_ph"]="checkpoint/can_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=0650-test_mean_score=1.000.ckpt"
    ["square_ph"]="checkpoint/square_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=3000-test_mean_score=1.000.ckpt"
    ["transport_ph"]="checkpoint/transport_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=3350-test_mean_score=0.955.ckpt"
    ["tool_hang_ph"]="checkpoint/tool_hang_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=1000-test_mean_score=0.682.ckpt"
    ["pusht"]="checkpoint/pusht/diffusion_policy_transformer/train_2/checkpoints/epoch=0150-test_mean_score=0.752.ckpt"
    ["lift_mh"]="checkpoint/lift_mh/diffusion_policy_transformer/train_2/checkpoints/epoch=0250-test_mean_score=1.000.ckpt"
    ["can_mh"]="checkpoint/can_mh/diffusion_policy_transformer/train_2/checkpoints/epoch=0500-test_mean_score=1.000.ckpt"
    ["square_mh"]="checkpoint/square_mh/diffusion_policy_transformer/train_2/checkpoints/epoch=3050-test_mean_score=0.955.ckpt"
    ["transport_mh"]="checkpoint/transport_mh/diffusion_policy_transformer/train_2/checkpoints/epoch=0350-test_mean_score=0.682.ckpt"
    ["block_pushing"]="checkpoint/low_dim/block_pushing/diffusion_policy_transformer/train_2/checkpoints/epoch=7950-test_mean_score=1.000.ckpt"
    ["kitchen"]="checkpoint/low_dim/kitchen/diffusion_policy_transformer/train_2/checkpoints/epoch=1750-test_mean_score=0.574.ckpt"
)

declare -A AVG_CHECKPOINTS_SEED0=(
    ["lift_ph"]="checkpoint/lift_ph/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["can_ph"]="checkpoint/can_ph/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["square_ph"]="checkpoint/square_ph/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["transport_ph"]="checkpoint/transport_ph/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["tool_hang_ph"]="checkpoint/tool_hang_ph/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["pusht"]="checkpoint/pusht/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["lift_mh"]="checkpoint/lift_mh/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["can_mh"]="checkpoint/can_mh/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["square_mh"]="checkpoint/square_mh/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["transport_mh"]="checkpoint/transport_mh/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["block_pushing"]="checkpoint/low_dim/block_pushing/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["kitchen"]="checkpoint/low_dim/kitchen/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
)

declare -A AVG_CHECKPOINTS_SEED1=(
    # PH
    ["lift_ph"]="checkpoint/lift_ph/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    ["can_ph"]="checkpoint/can_ph/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    ["square_ph"]="checkpoint/square_ph/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    ["transport_ph"]="checkpoint/transport_ph/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    ["pusht"]="checkpoint/pusht/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    
    # MH
    ["lift_mh"]="checkpoint/lift_mh/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    ["can_mh"]="checkpoint/can_mh/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    ["square_mh"]="checkpoint/square_mh/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    ["transport_mh"]="checkpoint/transport_mh/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    
    # LOWDIM
    ["block_pushing"]="checkpoint/low_dim/block_pushing/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    ["kitchen"]="checkpoint/low_dim/kitchen/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
)

declare -A AVG_CHECKPOINTS_SEED2=(
    # PH
    ["lift_ph"]="checkpoint/lift_ph/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["can_ph"]="checkpoint/can_ph/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["square_ph"]="checkpoint/square_ph/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["transport_ph"]="checkpoint/transport_ph/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["tool_hang_ph"]="checkpoint/tool_hang_ph/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["pusht"]="checkpoint/pusht/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    
    # MH
    ["lift_mh"]="checkpoint/lift_mh/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["can_mh"]="checkpoint/can_mh/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["square_mh"]="checkpoint/square_mh/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["transport_mh"]="checkpoint/transport_mh/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    
    # LOWDIM
    ["block_pushing"]="checkpoint/low_dim/block_pushing/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["kitchen"]="checkpoint/low_dim/kitchen/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
)

# 
usage() {
    echo ": $0 []"
    echo ":"
    echo "  --device device_name             (: ${DEVICE})"
    echo "  --task_type type                 (ph, mh, lowdim, all, both) (: both)"
    echo "  --tasks task1,task2,...          (: task_type)"
    echo "  --steps N                        (: 10)"
    echo "  --cache_mode mode                (threshold, original) (: threshold)"
    echo "  --seeds seed1,seed2,...          (: 0,1,2)"
    echo "  --output_dir dir                 (: )"
    echo "  --skip_video                     (: )"
    echo "  --with_video                     (skip_video)"
    echo "  --force                         ，"
    echo
    echo ":"
    echo "  $0 --device cuda:1 --task_type ph --tasks lift_ph,can_ph --steps 5 --seeds 0,1,2"
    echo "  $0 --device cuda:0 --task_type all --steps 10"
    echo "  $0 --device cuda:0 --task_type lowdim --with_video"
}

# 
print_title() {
    echo -e "\033[1;34m$1\033[0m"
}

# 
print_subtitle() {
    echo -e "\033[1;36m$1\033[0m"
}

# 
print_info() {
    echo -e "\033[0;32m$1\033[0m"
}

# 
print_warning() {
    echo -e "\033[0;33m$1\033[0m"
}

# 
print_error() {
    echo -e "\033[0;31m$1\033[0m"
}

# 
print_separator() {
    echo -e "\033[0;35m----------------------------------------\033[0m"
}

# 
while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --task_type)
            TASK_TYPE="$2"
            shift 2
            ;;
        --tasks)
            SPECIFIC_TASKS="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --cache_mode)
            CACHE_MODE="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip_video)
            SKIP_VIDEO="--skip_video"
            shift
            ;;
        --with_video)
            SKIP_VIDEO=""
            shift
            ;;
        --force)
            FORCE_RECOMPUTE=true
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo ": $1"
            usage
            exit 1
            ;;
    esac
done

# 
if [ -n "$SPECIFIC_TASKS" ]; then
    # ，
    IFS=',' read -ra ACTIVE_TASKS <<< "$SPECIFIC_TASKS"
else
    # 
    case "$TASK_TYPE" in
        "ph")
            ACTIVE_TASKS=("${PH_TASKS[@]}")
            ;;
        "mh")
            ACTIVE_TASKS=("${MH_TASKS[@]}")
            ;;
        "lowdim")
            ACTIVE_TASKS=("${LOWDIM_TASKS[@]}")
            ;;
        "both")
            ACTIVE_TASKS=("${PH_TASKS[@]}" "${MH_TASKS[@]}")
            ;;
        "all")
            ACTIVE_TASKS=("${PH_TASKS[@]}" "${MH_TASKS[@]}" "${LOWDIM_TASKS[@]}")
            ;;
        *)
            echo ":  '$TASK_TYPE',  'ph', 'mh', 'lowdim', 'all'  'both'"
            exit 1
            ;;
    esac
fi

# STEPSCACHE_THRESHOLD
CACHE_THRESHOLD=$((100 / STEPS))
if [ $CACHE_THRESHOLD -lt 1 ]; then
    CACHE_THRESHOLD=1
fi

# 
IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"

# 
if [ -z "$OUTPUT_DIR" ]; then
    # ，
    OUTPUT_DIR="results/benchmark/Uniform/${CACHE_MODE}_steps_${STEPS}"
fi

# 
mkdir -p "$OUTPUT_DIR"

# 
RESULTS_FILE="${OUTPUT_DIR}/results.csv"
echo "Method,Steps,Task,Seed,CheckpointType,SuccessRate,FLOPs,Speedup,TaskType" > "$RESULTS_FILE"

# 
print_title "====================================================="
print_title "                                      "
print_title "====================================================="
print_info ": $DEVICE"
print_info ": $TASK_TYPE"
print_info ": $STEPS"
print_info ": $CACHE_MODE"
if [ "$CACHE_MODE" = "threshold" ]; then
    print_info ": $CACHE_THRESHOLD (: 100/$STEPS)"
fi
print_info ": ${ACTIVE_TASKS[*]}"
print_info ": ${SEED_ARRAY[*]}"
print_info ": ${SKIP_VIDEO:+}"
print_info ": $OUTPUT_DIR"
print_info ": $FORCE_RECOMPUTE"
print_title "====================================================="
echo ""

# 
execute_evaluation() {
    local task_name="$1"
    local seed="$2"
    local checkpoint="$3"
    local checkpoint_type="$4"
    
    # 
    local task_type=""
    if [[ " ${PH_TASKS[*]} " =~ " ${task_name} " ]]; then
        task_type="ph"
    elif [[ " ${MH_TASKS[*]} " =~ " ${task_name} " ]]; then
        task_type="mh"
    elif [[ " ${LOWDIM_TASKS[*]} " =~ " ${task_name} " ]]; then
        task_type="lowdim"
    else
        print_warning ": $task_name"
        task_type="unknown"
    fi
    
    # 
    local task_output_dir="${OUTPUT_DIR}/${task_name}/seed${seed}_${checkpoint_type}"
    local result_file="${task_output_dir}/fast_eval_log.json"
    local metrics_file="${task_output_dir}/eval_results.json"
    
    # ，
    if [ -f "${result_file}" ] && [ -f "${metrics_file}" ] && [ "$FORCE_RECOMPUTE" = false ]; then
        print_info "，: ${task_output_dir}"
        
        # 
        if [[ "$task_name" == "block_pushing" ]]; then
            # Block Pushingp1p2
            local bp_p1=$(grep "\"test/p1\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local bp_p2=$(grep "\"test/p2\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local flops=$(grep "flops" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            local speedup=$(grep "speedup" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            
            # 
            echo "Unifrom,${STEPS},${task_name}_p1,${seed},${checkpoint_type},${bp_p1},${flops},${speedup},${task_type}" >> "$RESULTS_FILE"
            echo "Unifrom,${STEPS},${task_name}_p2,${seed},${checkpoint_type},${bp_p2},${flops},${speedup},${task_type}" >> "$RESULTS_FILE"
            
            print_info ": p1=${bp_p1}, p2=${bp_p2}, =${speedup}"
            
        elif [[ "$task_name" == "kitchen" ]]; then
            # Kitchenp_1p_4
            local kitchen_p1=$(grep "\"test/p_1\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local kitchen_p2=$(grep "\"test/p_2\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local kitchen_p3=$(grep "\"test/p_3\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local kitchen_p4=$(grep "\"test/p_4\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local flops=$(grep "flops" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            local speedup=$(grep "speedup" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            
            # 
            echo "Unifrom,${STEPS},${task_name}_p1,${seed},${checkpoint_type},${kitchen_p1},${flops},${speedup},${task_type}" >> "$RESULTS_FILE"
            echo "Unifrom,${STEPS},${task_name}_p2,${seed},${checkpoint_type},${kitchen_p2},${flops},${speedup},${task_type}" >> "$RESULTS_FILE"
            echo "Unifrom,${STEPS},${task_name}_p3,${seed},${checkpoint_type},${kitchen_p3},${flops},${speedup},${task_type}" >> "$RESULTS_FILE"
            echo "Unifrom,${STEPS},${task_name}_p4,${seed},${checkpoint_type},${kitchen_p4},${flops},${speedup},${task_type}" >> "$RESULTS_FILE"
            
            print_info ": p1=${kitchen_p1}, p2=${kitchen_p2}, p3=${kitchen_p3}, p4=${kitchen_p4}, =${speedup}"
        else
            # 
            local success_rate=$(grep "mean_score" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "0.0")
            local flops=$(grep "flops" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            local speedup=$(grep "speedup" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            
            # 
            echo "Unifrom,${STEPS},${task_name},${seed},${checkpoint_type},${success_rate},${flops},${speedup},${task_type}" >> "$RESULTS_FILE"
            
            print_info ": =${success_rate}, =${speedup}"
        fi
        
        print_separator
        return 0
    elif [ "$FORCE_RECOMPUTE" = true ]; then
        print_info "："
    fi
    
    # 
    mkdir -p "$task_output_dir"
    
    # 
    print_info ": ${task_name} (${task_type})"
    echo ": ${CACHE_MODE}, : ${STEPS}"
    if [ "$CACHE_MODE" = "threshold" ]; then
        echo ": ${CACHE_THRESHOLD} (: 100/$STEPS)"
    fi
    echo ": ${seed}"
    echo ": ${checkpoint}"
    echo ": ${checkpoint_type}"
    echo ": ${task_output_dir}"
    
    # (optimal)
    local optimal_steps_dir=""
    if [ "$CACHE_MODE" = "optimal" ]; then
        optimal_steps_dir="assets/${task_name}/original/optimal_steps/${METRIC}"
        
        # ，
        if [ ! -d "$optimal_steps_dir" ] || [ "$FORCE_RECOMPUTE" = true ]; then
            echo "，..."
            FORCE_FLAG=""
            if [ "$FORCE_RECOMPUTE" = true ]; then
                FORCE_FLAG="--force_recompute"
            fi
            python diffusion_policy/activation_utils/get_optimal_cache_update_steps.py \
                -c "$checkpoint" \
                -o "assets/${task_name}/original" \
                -d "$DEVICE" \
                --num_caches "$STEPS" \
                --metrics "cosine" \
                $FORCE_FLAG
        fi
    fi
    
    # 
    local eval_args=(
        --checkpoint "${checkpoint}"
        --output_dir "${task_output_dir}"
        --device "${DEVICE}"
        --cache_mode "${CACHE_MODE}"
    )
    
    # 
    if [ "$CACHE_MODE" = "threshold" ]; then
        eval_args+=(--cache_threshold "${CACHE_THRESHOLD}")
    elif [ "$CACHE_MODE" = "optimal" ]; then
        eval_args+=(
            --optimal_steps_dir "${optimal_steps_dir}"
            --metric "cosine"
            --num_caches "${STEPS}"
            --num_bu_blocks 5
        )
    fi
    
    # 
    if [ -n "$SKIP_VIDEO" ]; then
        eval_args+=($SKIP_VIDEO)
    fi
    
    # 
    python scripts/eval_fast_diffusion_policy.py "${eval_args[@]}"
    
    # 
    if [ -f "${result_file}" ]; then
        if [[ "$task_name" == "block_pushing" ]]; then
            # Block Pushingp1p2
            local bp_p1=$(grep "\"test/p1\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local bp_p2=$(grep "\"test/p2\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local flops=$(grep "flops" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            local speedup=$(grep "speedup" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            
            # 
            echo "Unifrom,${STEPS},${task_name}_p1,${seed},${checkpoint_type},${bp_p1},${flops},${speedup},${task_type}" >> "$RESULTS_FILE"
            echo "Unifrom,${STEPS},${task_name}_p2,${seed},${checkpoint_type},${bp_p2},${flops},${speedup},${task_type}" >> "$RESULTS_FILE"
            
            print_info ": ${task_name} p1=${bp_p1}, p2=${bp_p2}, =${speedup}"
            
        elif [[ "$task_name" == "kitchen" ]]; then
            # Kitchenp_1p_4
            local kitchen_p1=$(grep "\"test/p_1\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local kitchen_p2=$(grep "\"test/p_2\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local kitchen_p3=$(grep "\"test/p_3\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local kitchen_p4=$(grep "\"test/p_4\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local flops=$(grep "flops" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            local speedup=$(grep "speedup" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            
            # 
            echo "Unifrom,${STEPS},${task_name}_p1,${seed},${checkpoint_type},${kitchen_p1},${flops},${speedup},${task_type}" >> "$RESULTS_FILE"
            echo "Unifrom,${STEPS},${task_name}_p2,${seed},${checkpoint_type},${kitchen_p2},${flops},${speedup},${task_type}" >> "$RESULTS_FILE"
            echo "Unifrom,${STEPS},${task_name}_p3,${seed},${checkpoint_type},${kitchen_p3},${flops},${speedup},${task_type}" >> "$RESULTS_FILE"
            echo "Unifrom,${STEPS},${task_name}_p4,${seed},${checkpoint_type},${kitchen_p4},${flops},${speedup},${task_type}" >> "$RESULTS_FILE"
            
            print_info ": ${task_name} p1=${kitchen_p1}, p2=${kitchen_p2}, p3=${kitchen_p3}, p4=${kitchen_p4}, =${speedup}"
        else
            # 
            local success_rate=$(grep "mean_score" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "0.0")
            local flops=$(grep "flops" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            local speedup=$(grep "speedup" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            
            # 
            echo "Unifrom,${STEPS},${task_name},${seed},${checkpoint_type},${success_rate},${flops},${speedup},${task_type}" >> "$RESULTS_FILE"
            
            print_info ": =${success_rate}, =${speedup}"
        fi
    else
        print_warning ": : ${result_file}"
    fi
    
    print_separator
}

# 
for task_name in "${ACTIVE_TASKS[@]}"; do
    print_subtitle ": ${task_name}"
    
    for seed in "${SEED_ARRAY[@]}"; do
        for checkpoint_type in "max" "avg"; do
            # 
            if [ "$checkpoint_type" == "max" ]; then
                if [ "$seed" = "0" ]; then
                    checkpoint="${MAX_CHECKPOINTS_SEED0[$task_name]}"
                elif [ "$seed" = "1" ]; then
                    checkpoint="${MAX_CHECKPOINTS_SEED1[$task_name]}"
                elif [ "$seed" = "2" ]; then
                    checkpoint="${MAX_CHECKPOINTS_SEED2[$task_name]}"
                else
                    print_warning ": $seed, "
                    continue
                fi
            else
                if [ "$seed" = "0" ]; then
                    checkpoint="${AVG_CHECKPOINTS_SEED0[$task_name]}"
                elif [ "$seed" = "1" ]; then
                    checkpoint="${AVG_CHECKPOINTS_SEED1[$task_name]}"
                elif [ "$seed" = "2" ]; then
                    checkpoint="${AVG_CHECKPOINTS_SEED2[$task_name]}"
                else
                    print_warning ": $seed, "
                    continue
                fi
            fi
            
            # 
            if [ ! -f "$checkpoint" ]; then
                print_warning ": : $checkpoint, "
                continue
            fi
            
            # 
            execute_evaluation "$task_name" "$seed" "$checkpoint" "$checkpoint_type"
        done
    done
done

# 
print_title "..."
python - <<EOF
import pandas as pd
import numpy as np
import os

# 
results = pd.read_csv("$RESULTS_FILE")

# 
task_types = results['TaskType'].unique()

# 
for task_type in task_types:
    type_results = results[results['TaskType'] == task_type]
    
    print(f"\n\033[1;34m===== {task_type.upper()}  =====\033[0m")
    
    # 
    tasks = sorted(type_results['Task'].unique())
    
    # 
    print("\n\033[1;36m：\033[0m")
    for task in tasks:
        task_data = type_results[type_results['Task'] == task]
        print(f"\n: {task}")
        
        # 
        for ckpt_type in ["max", "avg"]:
            ckpt_data = task_data[task_data["CheckpointType"] == ckpt_type]
            if ckpt_data.empty:
                continue
                
            success_mean = ckpt_data['SuccessRate'].mean()
            success_std = ckpt_data['SuccessRate'].std()
            
            speedup_values = ckpt_data['Speedup'].replace("-", np.nan).astype(float)
            speedup_mean = speedup_values.mean()
            speedup_std = speedup_values.std()
            
            flops_values = ckpt_data['FLOPs'].replace("-", np.nan).astype(float)
            flops_mean = flops_values.mean()
            flops_std = flops_values.std()
            
            print(f"   {ckpt_type}:")
            print(f"    : {success_mean:.3f} ± {success_std:.3f}")
            
            if not np.isnan(speedup_mean):
                print(f"    : {speedup_mean:.2f}x ± {speedup_std:.2f}")
            else:
                print(f"    : -")
                
            if not np.isnan(flops_mean):
                print(f"    FLOPs: {flops_mean:.2f} ± {flops_std:.2f}")
            else:
                print(f"    FLOPs: -")
    
    # 
    print("\n\033[1;36m：\033[0m")
    for seed in sorted(type_results['Seed'].unique()):
        seed_data = type_results[type_results['Seed'] == seed]
        
        # 
        for ckpt_type in ["max", "avg"]:
            ckpt_seed_data = seed_data[seed_data["CheckpointType"] == ckpt_type]
            if ckpt_seed_data.empty:
                continue
                
            success_mean = ckpt_seed_data['SuccessRate'].mean()
            speedup_mean = ckpt_seed_data['Speedup'].replace("-", np.nan).astype(float).mean()
            
            print(f"   {seed},  {ckpt_type}:")
            print(f"    : {success_mean:.3f}")
            if not np.isnan(speedup_mean):
                print(f"    : {speedup_mean:.2f}x")
            else:
                print(f"    : -")
    
    # 
    overall_success = type_results['SuccessRate'].mean()
    overall_speedup = type_results['Speedup'].replace("-", np.nan).astype(float).mean()
    
    print(f"\n{task_type.upper()} :")
    print(f"  : {overall_success:.3f}")
    
    if not np.isnan(overall_speedup):
        print(f"  : {overall_speedup:.2f}x")
    else:
        print(f"  : -")

# 
overall_success = results['SuccessRate'].mean()
overall_speedup = results['Speedup'].replace("-", np.nan).astype(float).mean()

print(f"\n\033[1;34m=====  =====\033[0m")
print(f": {overall_success:.3f}")

if not np.isnan(overall_speedup):
    print(f": {overall_speedup:.2f}x")
else:
    print(f": -")

# LaTeX
print(f"\n\033[1;34m===== LaTeX =====\033[0m")
for task_type in task_types:
    type_results = results[results['TaskType'] == task_type]
    
    print(f"\n{task_type.upper()} :")
    
    # 
    for task in sorted(type_results['Task'].unique()):
        task_data = type_results[type_results['Task'] == task]
        
        # 
        max_success = task_data[task_data["CheckpointType"] == "max"]["SuccessRate"].mean()
        avg_success = task_data[task_data["CheckpointType"] == "avg"]["SuccessRate"].mean()
        
        # 
        speedup_values = task_data['Speedup'].replace("-", np.nan).astype(float)
        avg_speedup = speedup_values.mean()
        
        if np.isnan(avg_speedup):
            print(f"{task} & {max_success:.3f}/{avg_success:.3f} & - \\\\")
        else:
            print(f"{task} & {max_success:.3f}/{avg_success:.3f} & {avg_speedup:.2f}x \\\\")

# 
print(f"\n\033[1;34m=====  =====\033[0m")

# 
# 
task_mappings = {
    # PH
    "lift_ph": "Lift$_{ph}$",
    "can_ph": "Can$_{ph}$",
    "square_ph": "Square$_{ph}$",
    "transport_ph": "Trans$_{ph}$",
    "tool_hang_ph": "Tool$_{ph}$",
    "pusht": "Push--T",
    
    # MH
    "lift_mh": "Lift$_{mh}$",
    "can_mh": "Can$_{mh}$",
    "square_mh": "Square$_{mh}$",
    "transport_mh": "Trans$_{mh}$",
    
    # LOWDIM
    "block_pushing_p1": "BP$_{p1}$",
    "block_pushing_p2": "BP$_{p2}$",
    "kitchen_p1": "Kit$_{p1}$",
    "kitchen_p2": "Kit$_{p2}$",
    "kitchen_p3": "Kit$_{p3}$",
    "kitchen_p4": "Kit$_{p4}$"
}

# 
speedup_mean = results['Speedup'].replace("-", np.nan).astype(float).mean()
if np.isnan(speedup_mean):
    speedup_str = "--"
else:
    speedup_str = f"{speedup_mean:.2f}"

# 
for steps in results['Steps'].unique():
    step_results = results[results['Steps'] == steps]
    
    # PH
    ph_results = step_results[step_results['TaskType'] == 'ph']
    if not ph_results.empty:
        ph_data = {}
        for task in ['lift_ph', 'can_ph', 'square_ph', 'transport_ph', 'tool_hang_ph', 'pusht']:
            task_data = ph_results[ph_results['Task'] == task]
            if not task_data.empty:
                max_rate = task_data[task_data['CheckpointType'] == 'max']['SuccessRate'].mean()
                avg_rate = task_data[task_data['CheckpointType'] == 'avg']['SuccessRate'].mean()
                ph_data[task] = f"{max_rate:.2f}/{avg_rate:.2f}"
            else:
                ph_data[task] = "--/--"
        
        # PH
        ph_row = f"Uniform & {steps} & {ph_data.get('lift_ph', '--/--')} & {ph_data.get('can_ph', '--/--')} & {ph_data.get('square_ph', '--/--')} & {ph_data.get('transport_ph', '--/--')} & {ph_data.get('tool_hang_ph', '--/--')} & {ph_data.get('pusht', '--/--')} & -- & {speedup_str}"
        print("\n\033[1;36mPH (Uniform):\033[0m")
        print(ph_row)
    
    # MH
    mh_results = step_results[step_results['TaskType'] == 'mh']
    if not mh_results.empty:
        mh_data = {}
        for task in ['lift_mh', 'can_mh', 'square_mh', 'transport_mh']:
            task_data = mh_results[mh_results['Task'] == task]
            if not task_data.empty:
                max_rate = task_data[task_data['CheckpointType'] == 'max']['SuccessRate'].mean()
                avg_rate = task_data[task_data['CheckpointType'] == 'avg']['SuccessRate'].mean()
                mh_data[task] = f"{max_rate:.2f}/{avg_rate:.2f}"
            else:
                mh_data[task] = "--/--"
        
        # MH
        mh_speedup = mh_results['Speedup'].replace("-", np.nan).astype(float).mean()
        if np.isnan(mh_speedup):
            mh_speedup_str = "--"
        else:
            mh_speedup_str = f"{mh_speedup:.2f}"
            
        mh_row = f"Uniform & {steps} & {mh_data.get('lift_mh', '--/--')} & {mh_data.get('can_mh', '--/--')} & {mh_data.get('square_mh', '--/--')} & {mh_data.get('transport_mh', '--/--')} & -- & {mh_speedup_str}"
        print("\n\033[1;36mMH (Uniform):\033[0m")
        print(mh_row)
    
    # LOWDIM
    lowdim_results = step_results[step_results['TaskType'] == 'lowdim']
    if not lowdim_results.empty:
        lowdim_data = {}
        
        # block_pushingp1p2
        bp_data = lowdim_results[lowdim_results['Task'].str.contains('block_pushing')]
        if not bp_data.empty:
            bp_p1_data = bp_data[bp_data['Task'] == 'block_pushing_p1']
            if not bp_p1_data.empty:
                max_rate = bp_p1_data[bp_p1_data['CheckpointType'] == 'max']['SuccessRate'].mean()
                avg_rate = bp_p1_data[bp_p1_data['CheckpointType'] == 'avg']['SuccessRate'].mean()
                lowdim_data['block_pushing_p1'] = f"{max_rate:.2f}/{avg_rate:.2f}"
            
            bp_p2_data = bp_data[bp_data['Task'] == 'block_pushing_p2']
            if not bp_p2_data.empty:
                max_rate = bp_p2_data[bp_p2_data['CheckpointType'] == 'max']['SuccessRate'].mean()
                avg_rate = bp_p2_data[bp_p2_data['CheckpointType'] == 'avg']['SuccessRate'].mean()
                lowdim_data['block_pushing_p2'] = f"{max_rate:.2f}/{avg_rate:.2f}"
        
        # kitchenp1p4
        kitchen_data = lowdim_results[lowdim_results['Task'].str.contains('kitchen')]
        for i in range(1, 5):
            kitchen_pi_data = kitchen_data[kitchen_data['Task'] == f'kitchen_p{i}']
            if not kitchen_pi_data.empty:
                max_rate = kitchen_pi_data[kitchen_pi_data['CheckpointType'] == 'max']['SuccessRate'].mean()
                avg_rate = kitchen_pi_data[kitchen_pi_data['CheckpointType'] == 'avg']['SuccessRate'].mean()
                lowdim_data[f'kitchen_p{i}'] = f"{max_rate:.2f}/{avg_rate:.2f}"
            else:
                lowdim_data[f'kitchen_p{i}'] = "--/--"
        
        # LOWDIM
        lowdim_speedup = lowdim_results['Speedup'].replace("-", np.nan).astype(float).mean()
        if np.isnan(lowdim_speedup):
            lowdim_speedup_str = "--"
        else:
            lowdim_speedup_str = f"{lowdim_speedup:.2f}"
            
        lowdim_row = f"Uniform & {steps} & {lowdim_data.get('block_pushing_p1', '--/--')} & {lowdim_data.get('block_pushing_p2', '--/--')} & {lowdim_data.get('kitchen_p1', '--/--')} & {lowdim_data.get('kitchen_p2', '--/--')} & {lowdim_data.get('kitchen_p3', '--/--')} & {lowdim_data.get('kitchen_p4', '--/--')} & -- & {lowdim_speedup_str}"
        print("\n\033[1;36mLOWDIM (Uniform):\033[0m")
        print(lowdim_row)

# 
summary_file = "$RESULTS_FILE".replace(".csv", "_summary.csv")
summary_data = []

# 
for task in sorted(results['Task'].unique()):
    task_data = results[results['Task'] == task]
    task_type = task_data['TaskType'].iloc[0]
    
    for ckpt_type in ["max", "avg"]:
        ckpt_data = task_data[task_data["CheckpointType"] == ckpt_type]
        if ckpt_data.empty:
            continue
            
        success_mean = ckpt_data['SuccessRate'].mean()
        speedup_values = ckpt_data['Speedup'].replace("-", np.nan).astype(float)
        speedup_mean = speedup_values.mean()
        flops_values = ckpt_data['FLOPs'].replace("-", np.nan).astype(float)
        flops_mean = flops_values.mean()
        
        summary_data.append({
            "Task": task,
            "TaskType": task_type,
            "CheckpointType": ckpt_type,
            "Steps": $STEPS,
            "CacheMode": "${CACHE_MODE}",
            "SuccessRate": success_mean,
            "Speedup": speedup_mean if not np.isnan(speedup_mean) else "-",
            "FLOPs": flops_mean if not np.isnan(flops_mean) else "-"
        })

# 
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(summary_file, index=False)
print(f"\n: {summary_file}")
EOF

print_title "！:"
echo "- : $RESULTS_FILE"
echo "- : ${RESULTS_FILE//.csv/_summary.csv}"
echo "- : $OUTPUT_DIR/{task_name}/seed{seed}_{checkpoint_type}/"