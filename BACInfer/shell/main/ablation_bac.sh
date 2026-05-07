DEVICE="cuda:0"
SKIP_VIDEO="--skip_video"
METRIC="cosine"
TASK_TYPE="both" # PHMH，: "ph", "mh", "both", "lowdim", "all"
FORCE_RECOMPUTE=false # ，

# 
declare -a STEPS_CONFIGS=(10)


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

# 
declare -a LOWDIM_TASKS=(
    "block_pushing"
    "kitchen"
)

# （TASK_TYPE）
declare -a ACTIVE_TASKS=()

#  ()
declare -A MAX_CHECKPOINTS_SEED0=(
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
    # LOWDIM
    ["block_pushing"]="checkpoint/low_dim/block_pushing/diffusion_policy_transformer/train_0/checkpoints/epoch=7550-test_mean_score=1.000.ckpt"
    ["kitchen"]="checkpoint/low_dim/kitchen/diffusion_policy_transformer/train_0/checkpoints/epoch=3000-test_mean_score=0.574.ckpt"
)

declare -A MAX_CHECKPOINTS_SEED1=(
    ["lift_ph"]="checkpoint/lift_ph/diffusion_policy_transformer/train_1/checkpoints/epoch=0250-test_mean_score=1.000.ckpt"
    ["can_ph"]="checkpoint/can_ph/diffusion_policy_transformer/train_1/checkpoints/epoch=1150-test_mean_score=1.000.ckpt"
    ["square_ph"]="checkpoint/square_ph/diffusion_policy_transformer/train_1/checkpoints/epoch=2200-test_mean_score=1.000.ckpt"
    ["transport_ph"]="checkpoint/transport_ph/diffusion_policy_transformer/train_1/checkpoints/epoch=1800-test_mean_score=0.955.ckpt"
    ["tool_hang_ph"]="checkpoint/tool_hang_ph/diffusion_policy_transformer/train_1/checkpoints/epoch=2400-test_mean_score=0.682.ckpt"
    ["pusht"]="checkpoint/pusht/diffusion_policy_transformer/train_1/checkpoints/epoch=0400-test_mean_score=0.817.ckpt"
    ["lift_mh"]="checkpoint/lift_mh/diffusion_policy_transformer/train_1/checkpoints/epoch=0250-test_mean_score=1.000.ckpt"
    ["can_mh"]="checkpoint/can_mh/diffusion_policy_transformer/train_1/checkpoints/epoch=1100-test_mean_score=1.000.ckpt"
    ["square_mh"]="checkpoint/square_mh/diffusion_policy_transformer/train_1/checkpoints/epoch=2950-test_mean_score=0.864.ckpt"
    ["transport_mh"]="checkpoint/transport_mh/diffusion_policy_transformer/train_1/checkpoints/epoch=0200-test_mean_score=0.773.ckpt"
    # LOWDIM
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
    # LOWDIM
    ["block_pushing"]="checkpoint/low_dim/block_pushing/diffusion_policy_transformer/train_2/checkpoints/epoch=7950-test_mean_score=1.000.ckpt"
    ["kitchen"]="checkpoint/low_dim/kitchen/diffusion_policy_transformer/train_2/checkpoints/epoch=1750-test_mean_score=0.574.ckpt"
)

#  (10)
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
    # LOWDIM
    ["block_pushing"]="checkpoint/low_dim/block_pushing/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["kitchen"]="checkpoint/low_dim/kitchen/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
)

declare -A AVG_CHECKPOINTS_SEED1=(
    ["lift_ph"]="checkpoint/lift_ph/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    ["can_ph"]="checkpoint/can_ph/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    ["square_ph"]="checkpoint/square_ph/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    ["transport_ph"]="checkpoint/transport_ph/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    ["tool_hang_ph"]="checkpoint/tool_hang_ph/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    ["pusht"]="checkpoint/pusht/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    ["lift_mh"]="checkpoint/lift_mh/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    ["can_mh"]="checkpoint/can_mh/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    ["square_mh"]="checkpoint/square_mh/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    ["transport_mh"]="checkpoint/transport_mh/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    # LOWDIM
    ["block_pushing"]="checkpoint/low_dim/block_pushing/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    ["kitchen"]="checkpoint/low_dim/kitchen/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
)

declare -A AVG_CHECKPOINTS_SEED2=(
    ["lift_ph"]="checkpoint/lift_ph/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["can_ph"]="checkpoint/can_ph/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["square_ph"]="checkpoint/square_ph/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["transport_ph"]="checkpoint/transport_ph/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["tool_hang_ph"]="checkpoint/tool_hang_ph/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["pusht"]="checkpoint/pusht/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["lift_mh"]="checkpoint/lift_mh/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["can_mh"]="checkpoint/can_mh/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["square_mh"]="checkpoint/square_mh/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["transport_mh"]="checkpoint/transport_mh/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    # LOWDIM
    ["block_pushing"]="checkpoint/low_dim/block_pushing/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
    ["kitchen"]="checkpoint/low_dim/kitchen/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
)


# 
declare -a SEEDS=(0 1 2)

#  (，)
usage() {
    echo ": $0 []"
    echo ":"
    echo "  --device device_name             (: ${DEVICE})"
    echo "  --task_type type                 (ph, mh, both, lowdim, all) (: both)"
    echo "  --ph_tasks task1,task2,...      PH (: PH)"
    echo "  --mh_tasks task1,task2,...      MH (: MH)"
    echo "  --lowdim_tasks task1,task2,...   (: )"
    echo "  --steps steps1,steps2,...        (: 10)"
    echo "  --seeds seed1,seed2,...          (: 0,1,2)"
    echo "  --metric metric                  (: cosine)"
    echo "  --skip_video                    "
    echo "  --force                         ，"
    echo "  --bp_only                       block_pushing"
    echo "  --kitchen_only                  kitchen"
    echo
    echo ":"
    echo "  :"
    echo "  1. Unified ACS: edit，"
    echo "  2. Block-wise ACS: optimalnum_bu_blocks=0，"
    echo
    echo ":"
    echo "  $0 --device cuda:1 --task_type ph --ph_tasks lift_ph,can_ph --steps 5 --seeds 0 --skip_video"
    echo "  $0 --device cuda:0 --task_type both --steps 5,10 --metric cosine"
    echo "  $0 --device cuda:0 --task_type mh --mh_tasks lift_mh,can_mh --force"
    echo "  $0 --device cuda:0 --task_type lowdim --steps 5 --seeds 0 --skip_video"
    echo "  $0 --bp_only --device cuda:0 --steps 10 --metric mse"
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
        --ph_tasks)
            IFS=',' read -ra PH_TASKS <<< "$2"
            shift 2
            ;;
        --mh_tasks)
            IFS=',' read -ra MH_TASKS <<< "$2"
            shift 2
            ;;
        --lowdim_tasks)
            IFS=',' read -ra LOWDIM_TASKS <<< "$2"
            shift 2
            ;;
        --steps)
            IFS=',' read -ra STEPS_CONFIGS <<< "$2"
            shift 2
            ;;
        --seeds)
            IFS=',' read -ra SEEDS <<< "$2"
            shift 2
            ;;
        --metric)
            METRIC="$2"
            shift 2
            ;;
        --skip_video)
            SKIP_VIDEO="--skip_video"
            shift
            ;;
        --force)
            FORCE_RECOMPUTE=true
            shift
            ;;
        --bp_only)
            TASK_TYPE="block_pushing"
            shift
            ;;
        --kitchen_only)
            TASK_TYPE="kitchen"
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

# task_type
case "$TASK_TYPE" in
    "ph")
        ACTIVE_TASKS=("${PH_TASKS[@]}")
        ;;
    "mh")
        ACTIVE_TASKS=("${MH_TASKS[@]}")
        ;;
    "both")
        ACTIVE_TASKS=("${PH_TASKS[@]}" "${MH_TASKS[@]}")
        ;;
    "lowdim")
        ACTIVE_TASKS=("${LOWDIM_TASKS[@]}")
        ;;
    "all")
        ACTIVE_TASKS=("${PH_TASKS[@]}" "${MH_TASKS[@]}" "${LOWDIM_TASKS[@]}")
        ;;
    "block_pushing")
        ACTIVE_TASKS=("block_pushing")
        ;;
    "kitchen")
        ACTIVE_TASKS=("kitchen")
        ;;
    *)
        echo ":  '$TASK_TYPE',  'ph', 'mh', 'both', 'lowdim', 'all', 'block_pushing'  'kitchen'"
        exit 1
        ;;
esac

# 
BENCHMARK_RESULTS_DIR="results/ablation/bac"
mkdir -p "$BENCHMARK_RESULTS_DIR"

# 
OUTPUT_DIR_BASE="results/ablation/bac"
mkdir -p "$OUTPUT_DIR_BASE"

# 
RESULTS_FILE="${BENCHMARK_RESULTS_DIR}/results.csv"
PH_RESULTS_FILE="${BENCHMARK_RESULTS_DIR}/ph_results.csv"
MH_RESULTS_FILE="${BENCHMARK_RESULTS_DIR}/mh_results.csv"
BP_RESULTS_FILE="${BENCHMARK_RESULTS_DIR}/bp_results.csv"
KITCHEN_RESULTS_FILE="${BENCHMARK_RESULTS_DIR}/kitchen_results.csv"

# 
echo "Method,Steps,Task,Seed,CheckpointType,SuccessRate,FLOPs,Speedup,TaskType" > "$RESULTS_FILE"

# 
if [[ "$TASK_TYPE" == "ph" || "$TASK_TYPE" == "both" || "$TASK_TYPE" == "all" ]]; then
    echo "Method,Steps,Task,Seed,CheckpointType,SuccessRate,FLOPs,Speedup" > "$PH_RESULTS_FILE"
fi
if [[ "$TASK_TYPE" == "mh" || "$TASK_TYPE" == "both" || "$TASK_TYPE" == "all" ]]; then
    echo "Method,Steps,Task,Seed,CheckpointType,SuccessRate,FLOPs,Speedup" > "$MH_RESULTS_FILE"
fi
if [[ "$TASK_TYPE" == "lowdim" || "$TASK_TYPE" == "all" || "$TASK_TYPE" == "block_pushing" ]]; then
    echo "Method,Steps,Task,Seed,CheckpointType,SuccessRate,FLOPs,Speedup" > "$BP_RESULTS_FILE"
fi
if [[ "$TASK_TYPE" == "lowdim" || "$TASK_TYPE" == "all" || "$TASK_TYPE" == "kitchen" ]]; then
    echo "Method,Steps,Task,Seed,CheckpointType,SuccessRate,FLOPs,Speedup" > "$KITCHEN_RESULTS_FILE"
fi

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
# ，
execute_single_evaluation() {
    local method_name="$1"
    local cache_mode="$2"
    local cache_threshold="$3"  # threshold
    local edit_steps="$4"       # edit
    local optimal_steps_dir="$5" # optimal
    local num_caches="$6"        # optimal
    local num_bu_blocks="$7"     # optimal
    local task_name="$8"
    local task_type_name="$9"
    local seed="${10}"
    local checkpoint_type="${11}"
    local checkpoint="${12}"

    # 
    local cache_mode_var="${cache_mode}"
    local num_cache_var="${num_caches}"
    OUTPUT_DIR="${OUTPUT_DIR_BASE}/${method_name}/${cache_mode_var}_${METRIC}_caches${num_cache_var}_bu${num_bu_blocks}"
    mkdir -p "$OUTPUT_DIR"

    # 
    local task_output_dir="${OUTPUT_DIR}/${task_name}/seed${seed}_${checkpoint_type}"
    mkdir -p "$task_output_dir"
    local result_file="${task_output_dir}/fast_eval_log.json"
    local metrics_file="${task_output_dir}/eval_results.json"

    # ，
    if [ -f "${result_file}" ] && [ -f "${metrics_file}" ] && [ "$FORCE_RECOMPUTE" = false ]; then
        print_info "，: ${task_output_dir}"

        # 
        # block_pushing
        if [ "$task_name" = "block_pushing" ]; then
            # p1p2
            local bp_p1=$(grep "\"test/p1\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local bp_p2=$(grep "\"test/p2\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local flops=$(grep "flops" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            local speedup=$(grep "speedup" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")

            # 
            echo "${method_name},${num_caches},${task_name}_p1,${seed},${checkpoint_type},${bp_p1},${flops},${speedup},block_pushing" >> "$RESULTS_FILE"
            echo "${method_name},${num_caches},${task_name}_p2,${seed},${checkpoint_type},${bp_p2},${flops},${speedup},block_pushing" >> "$RESULTS_FILE"

            # Block-pushing
            echo "${method_name},${num_caches},${task_name}_p1,${seed},${checkpoint_type},${bp_p1},${flops},${speedup}" >> "$BP_RESULTS_FILE"
            echo "${method_name},${num_caches},${task_name}_p2,${seed},${checkpoint_type},${bp_p2},${flops},${speedup}" >> "$BP_RESULTS_FILE"

            print_info ":"
            echo "  BP P1: ${bp_p1}"
            echo "  BP P2: ${bp_p2}"
            echo "  FLOPs: ${flops}"
            echo "  Speedup: ${speedup}"

            print_separator
            return 0
        # kitchen
        elif [ "$task_name" = "kitchen" ]; then
            # p_1p_4
            local kitchen_p1=$(grep "\"test/p_1\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local kitchen_p2=$(grep "\"test/p_2\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local kitchen_p3=$(grep "\"test/p_3\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local kitchen_p4=$(grep "\"test/p_4\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local flops=$(grep "flops" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            local speedup=$(grep "speedup" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")

            # 
            echo "${method_name},${num_caches},${task_name}_p1,${seed},${checkpoint_type},${kitchen_p1},${flops},${speedup},kitchen" >> "$RESULTS_FILE"
            echo "${method_name},${num_caches},${task_name}_p2,${seed},${checkpoint_type},${kitchen_p2},${flops},${speedup},kitchen" >> "$RESULTS_FILE"
            echo "${method_name},${num_caches},${task_name}_p3,${seed},${checkpoint_type},${kitchen_p3},${flops},${speedup},kitchen" >> "$RESULTS_FILE"
            echo "${method_name},${num_caches},${task_name}_p4,${seed},${checkpoint_type},${kitchen_p4},${flops},${speedup},kitchen" >> "$RESULTS_FILE"

            # Kitchen
            echo "${method_name},${num_caches},${task_name}_p1,${seed},${checkpoint_type},${kitchen_p1},${flops},${speedup}" >> "$KITCHEN_RESULTS_FILE"
            echo "${method_name},${num_caches},${task_name}_p2,${seed},${checkpoint_type},${kitchen_p2},${flops},${speedup}" >> "$KITCHEN_RESULTS_FILE"
            echo "${method_name},${num_caches},${task_name}_p3,${seed},${checkpoint_type},${kitchen_p3},${flops},${speedup}" >> "$KITCHEN_RESULTS_FILE"
            echo "${method_name},${num_caches},${task_name}_p4,${seed},${checkpoint_type},${kitchen_p4},${flops},${speedup}" >> "$KITCHEN_RESULTS_FILE"

            print_info ":"
            echo "  Kitchen P1: ${kitchen_p1}"
            echo "  Kitchen P2: ${kitchen_p2}"
            echo "  Kitchen P3: ${kitchen_p3}"
            echo "  Kitchen P4: ${kitchen_p4}"
            echo "  FLOPs: ${flops}"
            echo "  Speedup: ${speedup}"

            print_separator
            return 0
        # 
        else
            local success_rate=$(grep "mean_score" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1)
            local flops=$(grep "flops" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            local speedup=$(grep "speedup" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")

            # 
            if [ -n "$success_rate" ] && [ "$success_rate" != "N/A" ]; then
                print_info ":"
                echo "  Success Rate: ${success_rate}"
                echo "  FLOPs: ${flops}"
                echo "  Speedup: ${speedup}"

                # 
                echo "${method_name},${num_caches},${task_name},${seed},${checkpoint_type},${success_rate},${flops},${speedup},${task_type_name}" >> "$RESULTS_FILE"

                # 
                if [ "$task_type_name" == "ph" ]; then
                    echo "${method_name},${num_caches},${task_name},${seed},${checkpoint_type},${success_rate},${flops},${speedup}" >> "$PH_RESULTS_FILE"
                elif [ "$task_type_name" == "mh" ]; then
                    echo "${method_name},${num_caches},${task_name},${seed},${checkpoint_type},${success_rate},${flops},${speedup}" >> "$MH_RESULTS_FILE"
                elif [ "$task_type_name" == "block_pushing" ]; then
                    echo "${method_name},${num_caches},${task_name},${seed},${checkpoint_type},${success_rate},${flops},${speedup}" >> "$BP_RESULTS_FILE"
                elif [ "$task_type_name" == "kitchen" ]; then
                    echo "${method_name},${num_caches},${task_name},${seed},${checkpoint_type},${success_rate},${flops},${speedup}" >> "$KITCHEN_RESULTS_FILE"
                fi

                print_separator
                return 0
            else
                print_warning "，"
            fi
        fi
    elif [ "$FORCE_RECOMPUTE" = true ]; then
        print_info "："
    fi

    # 
    print_info ": ${task_name} (${task_type_name})"
    if [ "$cache_mode" == "threshold" ]; then
        echo ": ${cache_mode}, : ${num_caches}, : ${cache_threshold}"
    elif [ "$cache_mode" == "edit" ]; then
        echo ": ${cache_mode}, : ${num_caches}"
    elif [ "$cache_mode" == "optimal" ]; then
        echo ": ${cache_mode}, : ${num_caches}, BU: ${num_bu_blocks}"
    else
        echo ": ${cache_mode}, : ${num_caches}"
    fi
    echo ": ${seed}, : ${checkpoint_type}"
    echo ": ${checkpoint}"
    echo ": ${task_output_dir}"

    # 
    local eval_args=(
        --checkpoint "${checkpoint}"
        --output_dir "${task_output_dir}"
        --device "${DEVICE}"
        --cache_mode "${cache_mode}"
    )

    # 
    if [ "$cache_mode" == "threshold" ] && [ -n "$cache_threshold" ]; then
        eval_args+=(--cache_threshold "${cache_threshold}")
    elif [ "$cache_mode" == "edit" ] && [ -n "$edit_steps" ]; then
        eval_args+=(--edit_steps "${edit_steps}")
    elif [ "$cache_mode" == "optimal" ] && [ -n "$optimal_steps_dir" ]; then
        eval_args+=(--optimal_steps_dir "${optimal_steps_dir}")
        eval_args+=(--num_caches "${num_caches}")
        eval_args+=(--metric "${METRIC}")
        eval_args+=(--num_bu_blocks "${num_bu_blocks}")
    fi

    # 
    if [ -n "$SKIP_VIDEO" ]; then
        eval_args+=($SKIP_VIDEO)
    fi

    # 
    python scripts/eval_fast_diffusion_policy.py "${eval_args[@]}"

    # 
    if [ -f "${result_file}" ]; then
        # block_pushing
        if [ "$task_name" = "block_pushing" ]; then
            # p1p2
            local bp_p1=$(grep "\"test/p1\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local bp_p2=$(grep "\"test/p2\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local flops=$(grep "flops" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            local speedup=$(grep "speedup" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")

            # 
            echo "${method_name},${num_caches},${task_name}_p1,${seed},${checkpoint_type},${bp_p1},${flops},${speedup},block_pushing" >> "$RESULTS_FILE"
            echo "${method_name},${num_caches},${task_name}_p2,${seed},${checkpoint_type},${bp_p2},${flops},${speedup},block_pushing" >> "$RESULTS_FILE"

            # Block-pushing
            echo "${method_name},${num_caches},${task_name}_p1,${seed},${checkpoint_type},${bp_p1},${flops},${speedup}" >> "$BP_RESULTS_FILE"
            echo "${method_name},${num_caches},${task_name}_p2,${seed},${checkpoint_type},${bp_p2},${flops},${speedup}" >> "$BP_RESULTS_FILE"

            print_info ": ${task_name} p1=${bp_p1}, p2=${bp_p2} (${method_name}, : ${num_caches}, : ${seed}, : ${checkpoint_type})"
        # kitchen
        elif [ "$task_name" = "kitchen" ]; then
            # p_1p_4
            local kitchen_p1=$(grep "\"test/p_1\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local kitchen_p2=$(grep "\"test/p_2\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local kitchen_p3=$(grep "\"test/p_3\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local kitchen_p4=$(grep "\"test/p_4\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local flops=$(grep "flops" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            local speedup=$(grep "speedup" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")

            # 
            echo "${method_name},${num_caches},${task_name}_p1,${seed},${checkpoint_type},${kitchen_p1},${flops},${speedup},kitchen" >> "$RESULTS_FILE"
            echo "${method_name},${num_caches},${task_name}_p2,${seed},${checkpoint_type},${kitchen_p2},${flops},${speedup},kitchen" >> "$RESULTS_FILE"
            echo "${method_name},${num_caches},${task_name}_p3,${seed},${checkpoint_type},${kitchen_p3},${flops},${speedup},kitchen" >> "$RESULTS_FILE"
            echo "${method_name},${num_caches},${task_name}_p4,${seed},${checkpoint_type},${kitchen_p4},${flops},${speedup},kitchen" >> "$RESULTS_FILE"

            # Kitchen
            echo "${method_name},${num_caches},${task_name}_p1,${seed},${checkpoint_type},${kitchen_p1},${flops},${speedup}" >> "$KITCHEN_RESULTS_FILE"
            echo "${method_name},${num_caches},${task_name}_p2,${seed},${checkpoint_type},${kitchen_p2},${flops},${speedup}" >> "$KITCHEN_RESULTS_FILE"
            echo "${method_name},${num_caches},${task_name}_p3,${seed},${checkpoint_type},${kitchen_p3},${flops},${speedup}" >> "$KITCHEN_RESULTS_FILE"
            echo "${method_name},${num_caches},${task_name}_p4,${seed},${checkpoint_type},${kitchen_p4},${flops},${speedup}" >> "$KITCHEN_RESULTS_FILE"

            print_info ": ${task_name} p1=${kitchen_p1}, p2=${kitchen_p2}, p3=${kitchen_p3}, p4=${kitchen_p4} (${method_name}, : ${num_caches}, : ${seed}, : ${checkpoint_type})"
        # 
        else
            # 
            local success_rate=$(grep "mean_score" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1)
            local flops=$(grep "flops" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            local speedup=$(grep "speedup" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")

            # 
            echo "${method_name},${num_caches},${task_name},${seed},${checkpoint_type},${success_rate},${flops},${speedup},${task_type_name}" >> "$RESULTS_FILE"

            # 
            if [ "$task_type_name" == "ph" ]; then
                echo "${method_name},${num_caches},${task_name},${seed},${checkpoint_type},${success_rate},${flops},${speedup}" >> "$PH_RESULTS_FILE"
            elif [ "$task_type_name" == "mh" ]; then
                echo "${method_name},${num_caches},${task_name},${seed},${checkpoint_type},${success_rate},${flops},${speedup}" >> "$MH_RESULTS_FILE"
            elif [ "$task_type_name" == "block_pushing" ]; then
                echo "${method_name},${num_caches},${task_name},${seed},${checkpoint_type},${success_rate},${flops},${speedup}" >> "$BP_RESULTS_FILE"
            elif [ "$task_type_name" == "kitchen" ]; then
                echo "${method_name},${num_caches},${task_name},${seed},${checkpoint_type},${success_rate},${flops},${speedup}" >> "$KITCHEN_RESULTS_FILE"
            fi

            print_info ": =${success_rate}, =${speedup}"
        fi
    else
        print_warning ": : ${result_file}"
    fi

    print_separator
}

# 
print_title "====================================================="
print_title "               BAC: Unified vs Block-wise     "
print_title "====================================================="
print_info ": $DEVICE"
print_info ": ${STEPS_CONFIGS[*]}"
print_info ": $TASK_TYPE"
print_info ": ${ACTIVE_TASKS[*]}"
print_info ": ${SEEDS[*]}"
print_info ": $METRIC"
print_info ": ${SKIP_VIDEO:+}"
print_info ": ${FORCE_RECOMPUTE}"
print_title "====================================================="
echo ""

# 
for STEPS in "${STEPS_CONFIGS[@]}"; do
    for TASK_NAME in "${ACTIVE_TASKS[@]}"; do
        # 
        if [[ "$TASK_NAME" == *"_ph"* ]] || [[ "$TASK_NAME" == "pusht" ]]; then
            TASK_TYPE_NAME="ph"
        elif [[ "$TASK_NAME" == *"_mh"* ]]; then
            TASK_TYPE_NAME="mh"
        elif [[ "$TASK_NAME" == "block_pushing" ]]; then
            TASK_TYPE_NAME="block_pushing"
        elif [[ "$TASK_NAME" == "kitchen" ]]; then
            TASK_TYPE_NAME="kitchen"
        else
            TASK_TYPE_NAME="unknown"
        fi

        print_subtitle ": ${TASK_NAME} (${TASK_TYPE_NAME}), : ${STEPS}"

        # 
        OPTIMAL_STEPS_DIR="assets/${TASK_NAME}/original/optimal_steps/${METRIC}"
        OPTIMAL_STEPS_FILE="${OPTIMAL_STEPS_DIR}/decoder.layers.0.self_attn/optimal_steps_decoder.layers.0.self_attn_${STEPS}_${METRIC}.pkl"
        EDIT_STEPS=""

        # （）
        if [ -f "$OPTIMAL_STEPS_FILE" ]; then
            # pkl
            EDIT_STEPS=$(python -c "
import pickle
with open('${OPTIMAL_STEPS_FILE}', 'rb') as f:
    steps = pickle.load(f)
print(','.join(map(str, steps)))
")
            print_info ": $EDIT_STEPS"
        else
            print_warning ": $OPTIMAL_STEPS_FILE"
            print_error ""
            print_separator
            continue
        fi

        for SEED in "${SEEDS[@]}"; do
            # 
            if [ "$SEED" == "0" ]; then
                MAX_CHECKPOINT="${MAX_CHECKPOINTS_SEED0[$TASK_NAME]}"
                AVG_CHECKPOINT="${AVG_CHECKPOINTS_SEED0[$TASK_NAME]}"
            elif [ "$SEED" == "1" ]; then
                MAX_CHECKPOINT="${MAX_CHECKPOINTS_SEED1[$TASK_NAME]}"
                AVG_CHECKPOINT="${AVG_CHECKPOINTS_SEED1[$TASK_NAME]}"
            else
                MAX_CHECKPOINT="${MAX_CHECKPOINTS_SEED2[$TASK_NAME]}"
                AVG_CHECKPOINT="${AVG_CHECKPOINTS_SEED2[$TASK_NAME]}"
            fi

            # 
            for CHECKPOINT_TYPE in "max" "avg"; do
                if [ "$CHECKPOINT_TYPE" == "max" ]; then
                    CHECKPOINT="$MAX_CHECKPOINT"
                else
                    CHECKPOINT="$AVG_CHECKPOINT"
                fi

                # 
                if [ ! -f "$CHECKPOINT" ]; then
                    print_warning ": : $CHECKPOINT, "
                    continue
                fi

                # Unified ACS (edit)
                execute_single_evaluation "Unified ACS" "edit" "" "$EDIT_STEPS" "" \
                    "$STEPS" "" "$TASK_NAME" "$TASK_TYPE_NAME" "$SEED" "$CHECKPOINT_TYPE" "$CHECKPOINT"

                # Block-wise ACS (optimal，num_bu_blocks=0)
                execute_single_evaluation "Block-wise ACS" "optimal" "" "" "$OPTIMAL_STEPS_DIR" \
                    "$STEPS" "0" "$TASK_NAME" "$TASK_TYPE_NAME" "$SEED" "$CHECKPOINT_TYPE" "$CHECKPOINT"
            done
        done
    done
done

# 
print_title "..."
python - <<EOF
import pandas as pd
import numpy as np
# 
results = pd.read_csv("$RESULTS_FILE")
# 
has_ph = any(results["TaskType"] == "ph")
has_mh = any(results["TaskType"] == "mh")
has_bp = any(results["TaskType"] == "block_pushing")
has_kitchen = any(results["TaskType"] == "kitchen")
# PH
if has_ph:
    print("\n\033[1;34m===== PH BAC =====\033[0m")
    ph_results = results[results["TaskType"] == "ph"]
    
    # LaTeX
    for method in sorted(ph_results["Method"].unique()):
        for steps in sorted(ph_results["Steps"].unique()):
            print(f"{method} & {steps} ", end="")
            
            # 
            ph_tasks = sorted(list(set([task for task in ph_results["Task"].unique() if task.endswith("_ph") or task == "pusht"])))
            for task in ph_tasks:
                # 
                task_data = ph_results[(ph_results["Method"] == method) & 
                                     (ph_results["Steps"] == steps) & 
                                     (ph_results["Task"] == task)]
                
                if task_data.empty:
                    print("& -/- ", end="")
                    continue
                    
                # 
                max_perf = task_data[task_data["CheckpointType"] == "max"]["SuccessRate"].mean()
                avg_perf = task_data[task_data["CheckpointType"] == "avg"]["SuccessRate"].mean()
                
                print(f"& {max_perf:.3f}/{avg_perf:.3f} ", end="")
            
            # FLOPs
            flops = ph_results[(ph_results["Method"] == method) & 
                             (ph_results["Steps"] == steps)]["FLOPs"].mean()
            speedup = ph_results[(ph_results["Method"] == method) & 
                             (ph_results["Steps"] == steps)]["Speedup"].mean()
            
            if np.isnan(flops):
                print(f"& - & {speedup:.2f}x \\\\\\\\")
            else:
                print(f"& {flops:.2f} & {speedup:.2f}x \\\\\\\\")
    
    # 
    print("\n\033[1;34m----- PH -----\033[0m")
    for method in sorted(ph_results["Method"].unique()):
        method_data = ph_results[ph_results["Method"] == method]
        avg_success = method_data["SuccessRate"].mean()
        avg_speedup = method_data["Speedup"].mean()
        print(f"{method}:  = {avg_success:.3f},  = {avg_speedup:.2f}x")
# MH
if has_mh:
    print("\n\033[1;34m===== MH BAC =====\033[0m")
    mh_results = results[results["TaskType"] == "mh"]
    
    # LaTeX
    for method in sorted(mh_results["Method"].unique()):
        for steps in sorted(mh_results["Steps"].unique()):
            print(f"{method} & {steps} ", end="")
            
            # 
            mh_tasks = sorted(list(set([task for task in mh_results["Task"].unique() if task.endswith("_mh")])))
            for task in mh_tasks:
                # 
                task_data = mh_results[(mh_results["Method"] == method) & 
                                     (mh_results["Steps"] == steps) & 
                                     (mh_results["Task"] == task)]
                
                if task_data.empty:
                    print("& -/- ", end="")
                    continue
                    
                # 
                max_perf = task_data[task_data["CheckpointType"] == "max"]["SuccessRate"].mean()
                avg_perf = task_data[task_data["CheckpointType"] == "avg"]["SuccessRate"].mean()
                
                print(f"& {max_perf:.3f}/{avg_perf:.3f} ", end="")
            
            # FLOPs
            flops = mh_results[(mh_results["Method"] == method) & 
                             (mh_results["Steps"] == steps)]["FLOPs"].mean()
            speedup = mh_results[(mh_results["Method"] == method) & 
                             (mh_results["Steps"] == steps)]["Speedup"].mean()
            
            if np.isnan(flops):
                print(f"& - & {speedup:.2f}x \\\\\\\\")
            else:
                print(f"& {flops:.2f} & {speedup:.2f}x \\\\\\\\")
    
    # 
    print("\n\033[1;34m----- MH -----\033[0m")
    for method in sorted(mh_results["Method"].unique()):
        method_data = mh_results[mh_results["Method"] == method]
        avg_success = method_data["SuccessRate"].mean()
        avg_speedup = method_data["Speedup"].mean()
        print(f"{method}:  = {avg_success:.3f},  = {avg_speedup:.2f}x")
# Block-pushing
if has_bp:
    print("\n\033[1;34m===== Block-pushing BAC =====\033[0m")
    bp_results = results[results["TaskType"] == "block_pushing"]
    
    # LaTeX
    for method in sorted(bp_results["Method"].unique()):
        for steps in sorted(bp_results["Steps"].unique()):
            print(f"{method} & {steps} ", end="")
            
            # p1p2
            bp_tasks = ["block_pushing_p1", "block_pushing_p2"]
            for task in bp_tasks:
                # 
                task_data = bp_results[(bp_results["Method"] == method) & 
                                     (bp_results["Steps"] == steps) & 
                                     (bp_results["Task"] == task)]
                
                if task_data.empty:
                    print("& -/- ", end="")
                    continue
                    
                # 
                max_perf = task_data[task_data["CheckpointType"] == "max"]["SuccessRate"].mean()
                avg_perf = task_data[task_data["CheckpointType"] == "avg"]["SuccessRate"].mean()
                
                print(f"& {max_perf:.3f}/{avg_perf:.3f} ", end="")
            
            # FLOPs
            flops = bp_results[(bp_results["Method"] == method) & 
                             (bp_results["Steps"] == steps)]["FLOPs"].mean()
            speedup = bp_results[(bp_results["Method"] == method) & 
                             (bp_results["Steps"] == steps)]["Speedup"].mean()
            
            if np.isnan(flops):
                print(f"& - & {speedup:.2f}x \\\\\\\\")
            else:
                print(f"& {flops:.2f} & {speedup:.2f}x \\\\\\\\")
    
    # 
    print("\n\033[1;34m----- Block-pushing -----\033[0m")
    for method in sorted(bp_results["Method"].unique()):
        method_data = bp_results[bp_results["Method"] == method]
        avg_success = method_data["SuccessRate"].mean()
        avg_speedup = method_data["Speedup"].mean()
        print(f"{method}:  = {avg_success:.3f},  = {avg_speedup:.2f}x")
# Kitchen
if has_kitchen:
    print("\n\033[1;34m===== Kitchen BAC =====\033[0m")
    kitchen_results = results[results["TaskType"] == "kitchen"]
    
    # LaTeX
    for method in sorted(kitchen_results["Method"].unique()):
        for steps in sorted(kitchen_results["Steps"].unique()):
            print(f"{method} & {steps} ", end="")
            
            # p1p4
            kitchen_tasks = ["kitchen_p1", "kitchen_p2", "kitchen_p3", "kitchen_p4"]
            for task in kitchen_tasks:
                # 
                task_data = kitchen_results[(kitchen_results["Method"] == method) & 
                                         (kitchen_results["Steps"] == steps) & 
                                         (kitchen_results["Task"] == task)]
                
                if task_data.empty:
                    print("& -/- ", end="")
                    continue
                    
                # 
                max_perf = task_data[task_data["CheckpointType"] == "max"]["SuccessRate"].mean()
                avg_perf = task_data[task_data["CheckpointType"] == "avg"]["SuccessRate"].mean()
                
                print(f"& {max_perf:.3f}/{avg_perf:.3f} ", end="")
            
            # FLOPs
            flops = kitchen_results[(kitchen_results["Method"] == method) & 
                                 (kitchen_results["Steps"] == steps)]["FLOPs"].mean()
            speedup = kitchen_results[(kitchen_results["Method"] == method) & 
                                  (kitchen_results["Steps"] == steps)]["Speedup"].mean()
            
            if np.isnan(flops):
                print(f"& - & {speedup:.2f}x \\\\\\\\")
            else:
                print(f"& {flops:.2f} & {speedup:.2f}x \\\\\\\\")
    
    # 
    print("\n\033[1;34m----- Kitchen -----\033[0m")
    for method in sorted(kitchen_results["Method"].unique()):
        method_data = kitchen_results[kitchen_results["Method"] == method]
        avg_success = method_data["SuccessRate"].mean()
        avg_speedup = method_data["Speedup"].mean()
        print(f"{method}:  = {avg_success:.3f},  = {avg_speedup:.2f}x")
# ，
if (has_ph and has_mh) or (has_ph and has_bp) or (has_ph and has_kitchen) or (has_mh and has_bp) or (has_mh and has_kitchen) or (has_bp and has_kitchen):
    print("\n\033[1;34m=====  =====\033[0m")
    for method in sorted(results["Method"].unique()):
        method_data = results[results["Method"] == method]
        avg_success = method_data["SuccessRate"].mean()
        avg_speedup = method_data["Speedup"].mean()
        
        # 
        task_type_stats = []
        if has_ph:
            ph_success = method_data[method_data["TaskType"] == "ph"]["SuccessRate"].mean()
            ph_speedup = method_data[method_data["TaskType"] == "ph"]["Speedup"].mean()
            task_type_stats.append(f"PH:  = {ph_success:.3f},  = {ph_speedup:.2f}x")
        
        if has_mh:
            mh_success = method_data[method_data["TaskType"] == "mh"]["SuccessRate"].mean()
            mh_speedup = method_data[method_data["TaskType"] == "mh"]["Speedup"].mean()
            task_type_stats.append(f"MH:  = {mh_success:.3f},  = {mh_speedup:.2f}x")
        
        if has_bp:
            bp_success = method_data[method_data["TaskType"] == "block_pushing"]["SuccessRate"].mean()
            bp_speedup = method_data[method_data["TaskType"] == "block_pushing"]["Speedup"].mean()
            task_type_stats.append(f"Block-pushing:  = {bp_success:.3f},  = {bp_speedup:.2f}x")
        
        if has_kitchen:
            kitchen_success = method_data[method_data["TaskType"] == "kitchen"]["SuccessRate"].mean()
            kitchen_speedup = method_data[method_data["TaskType"] == "kitchen"]["Speedup"].mean()
            task_type_stats.append(f"Kitchen:  = {kitchen_success:.3f},  = {kitchen_speedup:.2f}x")
        
        # 
        print(f"{method}:  = {avg_success:.3f},  = {avg_speedup:.2f}x")
        # 
        for stat in task_type_stats:
            print(f"  - {stat}")
EOF

# 
print_title "..."
for TASK_NAME in "${ACTIVE_TASKS[@]}"; do
    if [[ "$TASK_NAME" == *"_ph"* ]] || [[ "$TASK_NAME" == "pusht" ]]; then
        TASK_TYPE_NAME="ph"
    else
        TASK_TYPE_NAME="mh"
    fi

    TASK_SUMMARY_FILE="${BENCHMARK_RESULTS_DIR}/${TASK_NAME}_summary.csv"
    echo "Method,Steps,CacheMode,Metric,MaxCheckpointAvg,AvgCheckpointAvg,FLOPsAvg,SpeedupAvg" > "$TASK_SUMMARY_FILE"

    # awk
    awk -F, -v task="$TASK_NAME" -v outfile="$TASK_SUMMARY_FILE" '
    NR == 1 {next} # 
    $3 == task {
        method=$1
        steps=$2
        checkpoint=$5
        rate=$6
        flops=$7
        speedup=$8
        
        # 
        if (rate == "N/A" || rate == "-") next
        
        # 、
        if (checkpoint == "max") {
            max_sum[method,steps] += rate
            max_count[method,steps]++
        } else if (checkpoint == "avg") {
            avg_sum[method,steps] += rate
            avg_count[method,steps]++
        }
        
        # FLOPsSpeedup ()
        if (flops != "N/A" && flops != "-") {
            flops_sum[method,steps] += flops
            flops_count[method,steps]++
        }
        if (speedup != "N/A" && speedup != "-") {
            speedup_sum[method,steps] += speedup
            speedup_count[method,steps]++
        }
    }
    END {
        # 
        for (key in max_sum) {
            split(key, parts, SUBSEP)
            method = parts[1]
            steps = parts[2]
            
            # 
            max_avg = (max_count[key] > 0) ? max_sum[key]/max_count[key] : "N/A"
            avg_avg = (avg_count[key] > 0) ? avg_sum[key]/avg_count[key] : "N/A"
            flops_avg = (flops_count[key] > 0) ? flops_sum[key]/flops_count[key] : "N/A"
            speedup_avg = (speedup_count[key] > 0) ? speedup_sum[key]/speedup_count[key] : "N/A"
            
            # 
            printf "%s,%s,${cache_mode_var},${METRIC},%s,%s,%s,%s\n", method, steps, max_avg, avg_avg, flops_avg, speedup_avg >> outfile
        }
    }
    ' "$RESULTS_FILE"

    print_info " ${TASK_NAME} : ${TASK_SUMMARY_FILE}"
done

# 
FINAL_SUMMARY_FILE="${BENCHMARK_RESULTS_DIR}/final_summary.csv"
echo "Method,Steps,Task,TaskType,CacheMode,Metric,SuccessRate(MAX/AVG),FLOPs,Speedup" > "$FINAL_SUMMARY_FILE"

# 
for TASK_NAME in "${ACTIVE_TASKS[@]}"; do
    if [[ "$TASK_NAME" == *"_ph"* ]] || [[ "$TASK_NAME" == "pusht" ]]; then
        TASK_TYPE_NAME="ph"
    else
        TASK_TYPE_NAME="mh"
    fi

    TASK_SUMMARY_FILE="${BENCHMARK_RESULTS_DIR}/${TASK_NAME}_summary.csv"

    # ，
    if [ -f "$TASK_SUMMARY_FILE" ]; then
        tail -n +2 "$TASK_SUMMARY_FILE" | while IFS=, read -r method steps cache_mode metric max_avg avg_avg flops_avg speedup_avg; do
            # MAX/AVG
            if [ "$max_avg" = "N/A" ] && [ "$avg_avg" = "N/A" ]; then
                formatted_rate="-/-"
            elif [ "$max_avg" = "N/A" ]; then
                formatted_rate="-/${avg_avg}"
            elif [ "$avg_avg" = "N/A" ]; then
                formatted_rate="${max_avg}/-"
            else
                formatted_rate="${max_avg}/${avg_avg}"
            fi

            echo "${method},${steps},${TASK_NAME},${TASK_TYPE_NAME},${cache_mode},${metric},${formatted_rate},${flops_avg},${speedup_avg}" >> "$FINAL_SUMMARY_FILE"
        done
    fi
done

print_title "BAC！:"
echo "- : $RESULTS_FILE"
if [[ "$TASK_TYPE" == "ph" || "$TASK_TYPE" == "both" || "$TASK_TYPE" == "all" ]]; then
    echo "- PH: $PH_RESULTS_FILE"
fi
if [[ "$TASK_TYPE" == "mh" || "$TASK_TYPE" == "both" || "$TASK_TYPE" == "all" ]]; then
    echo "- MH: $MH_RESULTS_FILE"
fi
if [[ "$TASK_TYPE" == "lowdim" || "$TASK_TYPE" == "all" || "$TASK_TYPE" == "block_pushing" ]]; then
    echo "- Block-pushing: $BP_RESULTS_FILE"
fi
if [[ "$TASK_TYPE" == "lowdim" || "$TASK_TYPE" == "all" || "$TASK_TYPE" == "kitchen" ]]; then
    echo "- Kitchen: $KITCHEN_RESULTS_FILE"
fi
echo "- : $FINAL_SUMMARY_FILE"

# 
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    usage
    exit 0
fi
