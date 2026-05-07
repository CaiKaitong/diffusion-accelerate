DEVICE="cuda:3"
TASK_TYPE="all"  # : "ph", "mh", "lowdim", "all", "both" (ph+mh)
CACHE_MODE="optimal"
SKIP_VIDEO="--skip_video"
STEPS=10  # 
NUM_BU_BLOCKS=5  # BAC
METRIC="cosine"  # 
OUTPUT_DIR=""  # ，
SPECIFIC_TASKS=""  # ，
SEEDS="0,1,2"  # 
FORCE_RECOMPUTE=false  # ，
CHECKPOINT_TYPE="all"  # ，: "max", "avg", "all"

# 
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

# 
declare -a ACTIVE_TASKS=()

#  ()
declare -A MAX_CHECKPOINTS_SEED0=(
    # PH
    ["lift_ph"]="checkpoint/lift_ph/diffusion_policy_transformer/train_0/checkpoints/epoch=0250-test_mean_score=1.000.ckpt"
    ["can_ph"]="checkpoint/can_ph/diffusion_policy_transformer/train_0/checkpoints/epoch=0950-test_mean_score=1.000.ckpt"
    ["square_ph"]="checkpoint/square_ph/diffusion_policy_transformer/train_0/checkpoints/epoch=2400-test_mean_score=1.000.ckpt"
    ["transport_ph"]="checkpoint/transport_ph/diffusion_policy_transformer/train_0/checkpoints/epoch=2100-test_mean_score=0.955.ckpt"
    ["tool_hang_ph"]="checkpoint/tool_hang_ph/diffusion_policy_transformer/train_0/checkpoints/epoch=0100-test_mean_score=0.773.ckpt"
    ["pusht"]="checkpoint/pusht/diffusion_policy_transformer/train_0/checkpoints/epoch=0100-test_mean_score=0.748.ckpt"
    
    # MH
    ["lift_mh"]="checkpoint/lift_mh/diffusion_policy_transformer/train_0/checkpoints/epoch=0250-test_mean_score=1.000.ckpt"
    ["can_mh"]="checkpoint/can_mh/diffusion_policy_transformer/train_0/checkpoints/epoch=1500-test_mean_score=1.000.ckpt"
    ["square_mh"]="checkpoint/square_mh/diffusion_policy_transformer/train_0/checkpoints/epoch=3050-test_mean_score=1.000.ckpt"
    ["transport_mh"]="checkpoint/transport_mh/diffusion_policy_transformer/train_0/checkpoints/epoch=1750-test_mean_score=0.727.ckpt"
    
    # LOWDIM
    ["block_pushing"]="checkpoint/low_dim/block_pushing/diffusion_policy_transformer/train_0/checkpoints/epoch=7550-test_mean_score=1.000.ckpt"
    ["kitchen"]="checkpoint/low_dim/kitchen/diffusion_policy_transformer/train_0/checkpoints/epoch=3000-test_mean_score=0.574.ckpt"
)

declare -A MAX_CHECKPOINTS_SEED1=(
    # PH
    ["lift_ph"]="checkpoint/lift_ph/diffusion_policy_transformer/train_1/checkpoints/epoch=0250-test_mean_score=1.000.ckpt"
    ["can_ph"]="checkpoint/can_ph/diffusion_policy_transformer/train_1/checkpoints/epoch=1150-test_mean_score=1.000.ckpt"
    ["square_ph"]="checkpoint/square_ph/diffusion_policy_transformer/train_1/checkpoints/epoch=2200-test_mean_score=1.000.ckpt"
    ["transport_ph"]="checkpoint/transport_ph/diffusion_policy_transformer/train_1/checkpoints/epoch=1800-test_mean_score=0.955.ckpt"
    ["tool_hang_ph"]="checkpoint/tool_hang_ph/diffusion_policy_transformer/train_1/checkpoints/epoch=2400-test_mean_score=0.682.ckpt"
    ["pusht"]="checkpoint/pusht/diffusion_policy_transformer/train_1/checkpoints/epoch=0400-test_mean_score=0.817.ckpt"
    
    # MH
    ["lift_mh"]="checkpoint/lift_mh/diffusion_policy_transformer/train_1/checkpoints/epoch=0250-test_mean_score=1.000.ckpt"
    ["can_mh"]="checkpoint/can_mh/diffusion_policy_transformer/train_1/checkpoints/epoch=1100-test_mean_score=1.000.ckpt"
    ["square_mh"]="checkpoint/square_mh/diffusion_policy_transformer/train_1/checkpoints/epoch=2950-test_mean_score=0.864.ckpt"
    ["transport_mh"]="checkpoint/transport_mh/diffusion_policy_transformer/train_1/checkpoints/epoch=0200-test_mean_score=0.773.ckpt"
    
    # LOWDIM
    ["block_pushing"]="checkpoint/low_dim/block_pushing/diffusion_policy_transformer/train_1/checkpoints/epoch=7950-test_mean_score=1.000.ckpt"
    ["kitchen"]="checkpoint/low_dim/kitchen/diffusion_policy_transformer/train_1/checkpoints/epoch=2700-test_mean_score=0.574.ckpt"
)

declare -A MAX_CHECKPOINTS_SEED2=(
    # PH
    ["lift_ph"]="checkpoint/lift_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=0250-test_mean_score=1.000.ckpt"
    ["can_ph"]="checkpoint/can_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=0650-test_mean_score=1.000.ckpt"
    ["square_ph"]="checkpoint/square_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=3000-test_mean_score=1.000.ckpt"
    ["transport_ph"]="checkpoint/transport_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=3350-test_mean_score=0.955.ckpt"
    ["tool_hang_ph"]="checkpoint/tool_hang_ph/diffusion_policy_transformer/train_2/checkpoints/epoch=1000-test_mean_score=0.682.ckpt"
    ["pusht"]="checkpoint/pusht/diffusion_policy_transformer/train_2/checkpoints/epoch=0150-test_mean_score=0.752.ckpt"
    
    # MH
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
    # PH
    ["lift_ph"]="checkpoint/lift_ph/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["can_ph"]="checkpoint/can_ph/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["square_ph"]="checkpoint/square_ph/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["transport_ph"]="checkpoint/transport_ph/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["tool_hang_ph"]="checkpoint/tool_hang_ph/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["pusht"]="checkpoint/pusht/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    
    # MH
    ["lift_mh"]="checkpoint/lift_mh/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["can_mh"]="checkpoint/can_mh/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["square_mh"]="checkpoint/square_mh/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["transport_mh"]="checkpoint/transport_mh/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    
    # LOWDIM
    ["block_pushing"]="checkpoint/low_dim/block_pushing/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
    ["kitchen"]="checkpoint/low_dim/kitchen/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
)

declare -A AVG_CHECKPOINTS_SEED1=(
    # PH
    ["lift_ph"]="checkpoint/lift_ph/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    ["can_ph"]="checkpoint/can_ph/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    ["square_ph"]="checkpoint/square_ph/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    ["transport_ph"]="checkpoint/transport_ph/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
    ["tool_hang_ph"]="checkpoint/tool_hang_ph/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
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
    echo "  --device device_name          (: ${DEVICE})"
    echo "  --steps N                     (: 10)"
    echo "  --seeds seed1,seed2,...       (: 0,1,2)"
    echo "  --metric metric_name          (: cosine)"
    echo "  --cache_mode mode             (optimal, threshold) (: optimal)"
    echo "  --skip_video                  (: )"
    echo "  --output_dir dir              (: )"
    echo "  --force                      ，"
    echo "  --checkpoint_type type        (avg, max, all) (: avg)"
    echo "                               'all'avgmax"
    echo "  --task task1,task2,...       "
    echo "  --ph_only                    PH"
    echo "  --mh_only                    MH"
    echo "  --lowdim_only                 (block_pushingkitchen)"
    echo "  --bp_only                    block_pushing"
    echo "  --kitchen_only               kitchen"
    echo
    echo ":"
    echo "  $0 --device cuda:0 --steps 5 --seeds 0,1,2 --metric l1"
    echo "  $0 --device cuda:0 --ph_only --checkpoint_type max"
    echo "  $0 --device cuda:0 --task lift_ph,can_ph --checkpoint_type all"
}

# 
while [ "$#" -gt 0 ]; do
    case "$1" in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --metric)
            METRIC="$2"
            shift 2
            ;;
        --cache_mode)
            CACHE_MODE="$2"
            shift 2
            ;;
        --num_bu_blocks)
            NUM_BU_BLOCKS="$2"
            shift 2
            ;;
        --skip_video)
            SKIP_VIDEO="--skip_video"
            shift
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --task)
            SPECIFIC_TASKS="$2"
            shift 2
            ;;
        --force)
            FORCE_RECOMPUTE=true
            shift
            ;;
        --checkpoint_type)
            CHECKPOINT_TYPE="$2"
            # 
            if [ "$CHECKPOINT_TYPE" != "max" ] && [ "$CHECKPOINT_TYPE" != "avg" ] && [ "$CHECKPOINT_TYPE" != "all" ]; then
                print_error ": $CHECKPOINT_TYPE"
                print_error ": max, avg, all"
                exit 1
            fi
            shift 2
            ;;
        --ph_only)
            TASK_TYPE="ph"
            shift
            ;;
        --mh_only)
            TASK_TYPE="mh"
            shift
            ;;
        --lowdim_only)
            TASK_TYPE="lowdim"
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

# 
if [ -z "$OUTPUT_DIR" ]; then
    # ，
    OUTPUT_DIR="results/benchmark/BAC/${CACHE_MODE}_${METRIC}_caches${STEPS}_bu${NUM_BU_BLOCKS}"
fi

# 
mkdir -p "$OUTPUT_DIR"

# RESULTS_DIR（）
RESULTS_DIR="$OUTPUT_DIR"
RESULTS_FILE="${RESULTS_DIR}/results.csv"

# 
echo "Method,Steps,Task,Seed,CheckpointType,SuccessRate,FLOPs,Speedup" > "$RESULTS_FILE"

# 
PH_RESULTS_FILE="${RESULTS_DIR}/ph_results.csv"
MH_RESULTS_FILE="${RESULTS_DIR}/mh_results.csv"
BP_RESULTS_FILE="${RESULTS_DIR}/bp_results.csv"
KITCHEN_RESULTS_FILE="${RESULTS_DIR}/kitchen_results.csv"

# 
echo "Method,Steps,Task,Seed,CheckpointType,SuccessRate,FLOPs,Speedup" > "$PH_RESULTS_FILE"
echo "Method,Steps,Task,Seed,CheckpointType,SuccessRate,FLOPs,Speedup" > "$MH_RESULTS_FILE"
echo "Method,Steps,Task,Seed,CheckpointType,SuccessRate,FLOPs,Speedup" > "$BP_RESULTS_FILE"
echo "Method,Steps,Task,Seed,CheckpointType,SuccessRate,FLOPs,Speedup" > "$KITCHEN_RESULTS_FILE"

# 
if [ "$TASK_TYPE" = "all" ]; then
    ACTIVE_TASKS=("${PH_TASKS[@]}" "${MH_TASKS[@]}" "${LOWDIM_TASKS[@]}")
elif [ "$TASK_TYPE" = "both" ]; then
    ACTIVE_TASKS=("${PH_TASKS[@]}" "${MH_TASKS[@]}")
elif [ "$TASK_TYPE" = "ph" ]; then
    ACTIVE_TASKS=("${PH_TASKS[@]}")
elif [ "$TASK_TYPE" = "mh" ]; then
    ACTIVE_TASKS=("${MH_TASKS[@]}")
elif [ "$TASK_TYPE" = "lowdim" ]; then
    ACTIVE_TASKS=("${LOWDIM_TASKS[@]}")
elif [ "$TASK_TYPE" = "block_pushing" ]; then
    ACTIVE_TASKS=("block_pushing")
elif [ "$TASK_TYPE" = "kitchen" ]; then
    ACTIVE_TASKS=("kitchen")
fi

# ，
if [ ! -z "$SPECIFIC_TASKS" ]; then
    IFS=',' read -ra SPECIFIC_TASKS_ARRAY <<< "$SPECIFIC_TASKS"
    ACTIVE_TASKS=("${SPECIFIC_TASKS_ARRAY[@]}")
fi

# 
echo "====================================================="
echo "BAC"
echo "====================================================="
echo ": $DEVICE"
echo ": $STEPS"
echo ": $SEEDS"
echo ": $METRIC"
echo ": ${SKIP_VIDEO:+}"
echo ": $TASK_TYPE"
echo ": $CACHE_MODE"
echo ": $NUM_BU_BLOCKS"
echo ": $OUTPUT_DIR"
echo ": $CHECKPOINT_TYPE"
echo ": $FORCE_RECOMPUTE"
echo ": ${ACTIVE_TASKS[*]}"
echo "====================================================="

# 
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # 

# 
print_title() {
    echo -e "${RED}$1${NC}"
}

# 
print_subtitle() {
    echo -e "${BLUE}$1${NC}"
}

# 
print_info() {
    echo -e "${GREEN}$1${NC}"
}

# 
print_warning() {
    echo -e "${YELLOW}$1${NC}"
}

# 
print_error() {
    echo -e "${RED}$1${NC}"
}

# 
print_separator() {
    echo -e "${BLUE}----------------------------------------${NC}"
}

# 
execute_single_evaluation() {
    local task_name="$1"
    local seed="$2"
    local checkpoint_type="$3"
    
    # 
    local task_output_dir="${OUTPUT_DIR}/${task_name}/seed${seed}_${checkpoint_type}"
    local result_file="${task_output_dir}/fast_eval_log.json"
    local metrics_file="${task_output_dir}/eval_results.json"
    
    # ，
    if [ -f "${result_file}" ] && [ -f "${metrics_file}" ] && [ "$FORCE_RECOMPUTE" = false ]; then
        print_info "，: ${task_output_dir}"
        
        # 
        # block_pushingkitchen，
        if [ "$task_name" = "block_pushing" ]; then
            # p1p2
            local BP_P1=$(grep "\"test/p1\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local BP_P2=$(grep "\"test/p2\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local SPEEDUP=$(grep "speedup" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            local FLOPS=$(grep "flops" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            
            # 
            echo "BAC,${STEPS},${task_name}_p1,${seed},${checkpoint_type},${BP_P1},${FLOPS},${SPEEDUP}" >> "$RESULTS_FILE"
            echo "BAC,${STEPS},${task_name}_p2,${seed},${checkpoint_type},${BP_P2},${FLOPS},${SPEEDUP}" >> "$RESULTS_FILE"
            
            # 
            echo "BAC,${STEPS},${task_name}_p1,${seed},${checkpoint_type},${BP_P1},${FLOPS},${SPEEDUP}" >> "$BP_RESULTS_FILE"
            echo "BAC,${STEPS},${task_name}_p2,${seed},${checkpoint_type},${BP_P2},${FLOPS},${SPEEDUP}" >> "$BP_RESULTS_FILE"
            
            print_info ": ${task_name} p1=${BP_P1}, p2=${BP_P2} (BAC, : ${STEPS}, : ${seed}, : ${checkpoint_type})"
        elif [ "$task_name" = "kitchen" ]; then
            # p_1p_4
            local KITCHEN_P1=$(grep "\"test/p_1\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local KITCHEN_P2=$(grep "\"test/p_2\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local KITCHEN_P3=$(grep "\"test/p_3\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local KITCHEN_P4=$(grep "\"test/p_4\":" "${result_file}" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local SPEEDUP=$(grep "speedup" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            local FLOPS=$(grep "flops" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            
            # 
            echo "BAC,${STEPS},${task_name}_p1,${seed},${checkpoint_type},${KITCHEN_P1},${FLOPS},${SPEEDUP}" >> "$RESULTS_FILE"
            echo "BAC,${STEPS},${task_name}_p2,${seed},${checkpoint_type},${KITCHEN_P2},${FLOPS},${SPEEDUP}" >> "$RESULTS_FILE"
            echo "BAC,${STEPS},${task_name}_p3,${seed},${checkpoint_type},${KITCHEN_P3},${FLOPS},${SPEEDUP}" >> "$RESULTS_FILE"
            echo "BAC,${STEPS},${task_name}_p4,${seed},${checkpoint_type},${KITCHEN_P4},${FLOPS},${SPEEDUP}" >> "$RESULTS_FILE"
            
            # 
            echo "BAC,${STEPS},${task_name}_p1,${seed},${checkpoint_type},${KITCHEN_P1},${FLOPS},${SPEEDUP}" >> "$KITCHEN_RESULTS_FILE"
            echo "BAC,${STEPS},${task_name}_p2,${seed},${checkpoint_type},${KITCHEN_P2},${FLOPS},${SPEEDUP}" >> "$KITCHEN_RESULTS_FILE"
            echo "BAC,${STEPS},${task_name}_p3,${seed},${checkpoint_type},${KITCHEN_P3},${FLOPS},${SPEEDUP}" >> "$KITCHEN_RESULTS_FILE"
            echo "BAC,${STEPS},${task_name}_p4,${seed},${checkpoint_type},${KITCHEN_P4},${FLOPS},${SPEEDUP}" >> "$KITCHEN_RESULTS_FILE"
            
            print_info ": ${task_name} p1=${KITCHEN_P1}, p2=${KITCHEN_P2}, p3=${KITCHEN_P3}, p4=${KITCHEN_P4} (BAC, : ${STEPS}, : ${seed}, : ${checkpoint_type})"
        else
            # 
            local SUCCESS_RATE=$(grep "mean_score" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "0.0")
            local SPEEDUP=$(grep "speedup" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            local FLOPS=$(grep "flops" "${metrics_file}" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            
            # 
            echo "BAC,${STEPS},${task_name},${seed},${checkpoint_type},${SUCCESS_RATE},${FLOPS},${SPEEDUP}" >> "$RESULTS_FILE"
            
            # 
            if [[ " ${PH_TASKS[*]} " =~ " ${task_name} " ]]; then
                echo "BAC,${STEPS},${task_name},${seed},${checkpoint_type},${SUCCESS_RATE},${FLOPS},${SPEEDUP}" >> "$PH_RESULTS_FILE"
            elif [[ " ${MH_TASKS[*]} " =~ " ${task_name} " ]]; then
                echo "BAC,${STEPS},${task_name},${seed},${checkpoint_type},${SUCCESS_RATE},${FLOPS},${SPEEDUP}" >> "$MH_RESULTS_FILE"
            fi
            
            print_info ": ${task_name} (BAC, : ${STEPS}, : ${seed}, : ${checkpoint_type}, : ${SUCCESS_RATE})"
        fi
        
        print_separator
        return 0
    fi
    
    # 
    local checkpoint=""
    if [ "$checkpoint_type" = "max" ]; then
        if [ "$seed" = "0" ]; then
            checkpoint="${MAX_CHECKPOINTS_SEED0[$task_name]}"
        elif [ "$seed" = "1" ]; then
            checkpoint="${MAX_CHECKPOINTS_SEED1[$task_name]}"
        elif [ "$seed" = "2" ]; then
            checkpoint="${MAX_CHECKPOINTS_SEED2[$task_name]}"
        fi
    else # avg
        if [ "$seed" = "0" ]; then
            checkpoint="${AVG_CHECKPOINTS_SEED0[$task_name]}"
        elif [ "$seed" = "1" ]; then
            checkpoint="${AVG_CHECKPOINTS_SEED1[$task_name]}"
        elif [ "$seed" = "2" ]; then
            checkpoint="${AVG_CHECKPOINTS_SEED2[$task_name]}"
        fi
    fi
    
    if [ -z "$checkpoint" ]; then
        print_warning " $task_name  $seed ，"
        return 1
    fi
    
    # 
    local OPTIMAL_STEPS_DIR="assets/${task_name}/original/optimal_steps/${METRIC}"
    
    # 
    print_info ": =${STEPS}, =${task_name}, =${seed}, =${checkpoint_type}"
    
    # 
    if [ ! -d "$OPTIMAL_STEPS_DIR" ]; then
        print_info "，..."
        FORCE_FLAG=""
        if [ "$FORCE_RECOMPUTE" = true ]; then
            FORCE_FLAG="--force_recompute"
        fi
        python -m BACInfer.analysis.get_optimal_cache_update_steps \
            -c "$checkpoint" \
            -o "assets/${task_name}/original" \
            -d "$DEVICE" \
            --num_caches "$STEPS" \
            --metrics "$METRIC" \
            $FORCE_FLAG
    fi
    
    # 
    if [ ! -d "$OPTIMAL_STEPS_DIR" ]; then
        print_error ": : $OPTIMAL_STEPS_DIR"
        print_error " ${task_name} ，..."
        return 1
    fi
    
    # 
    local step_files_count=$(find "$OPTIMAL_STEPS_DIR" -name "*_${STEPS}_${METRIC}.pkl" 2>/dev/null | wc -l)
    if [ "$step_files_count" -eq 0 ]; then
        print_error ": ${STEPS}: $OPTIMAL_STEPS_DIR/*_${STEPS}_${METRIC}.pkl"
        print_error " ${task_name} ，..."
        return 1
    fi
    
    # BAC
    local task_output_dir="${OUTPUT_DIR}/${task_name}/seed${seed}_${checkpoint_type}"
    
    print_info ": ${task_name}, : ${CACHE_MODE}, : ${STEPS}, : ${seed}, : ${checkpoint_type}"
    print_info ": ${checkpoint}"
    print_info ": ${task_output_dir}"
    print_info ": ${OPTIMAL_STEPS_DIR}"
    
    # ，
    if [ -f "${task_output_dir}/fast_eval_log.json" ] && [ -f "${task_output_dir}/eval_results.json" ] && [ "$FORCE_RECOMPUTE" = false ]; then
        print_warning "，: ${task_output_dir}"
    else
        # 
        local eval_args=(
            --checkpoint "${checkpoint}"
            --output_dir "${task_output_dir}"
            --device "${DEVICE}"
            --cache_mode "${CACHE_MODE}"
            --optimal_steps_dir "${OPTIMAL_STEPS_DIR}"
            --metric "${METRIC}"
            --num_caches "${STEPS}"
            --num_bu_blocks "${NUM_BU_BLOCKS}"
            ${SKIP_VIDEO}
        )
        python -m BACInfer.scripts.eval_fast_diffusion_policy "${eval_args[@]}"
    fi
    
    # 
    local TASK_TYPE_FILE=""
    if [[ " ${PH_TASKS[*]} " =~ " ${task_name} " ]]; then
        TASK_TYPE_FILE="$PH_RESULTS_FILE"
    elif [[ " ${MH_TASKS[*]} " =~ " ${task_name} " ]]; then
        TASK_TYPE_FILE="$MH_RESULTS_FILE"
    elif [ "$task_name" = "block_pushing" ]; then
        TASK_TYPE_FILE="$BP_RESULTS_FILE"
    elif [ "$task_name" = "kitchen" ]; then
        TASK_TYPE_FILE="$KITCHEN_RESULTS_FILE"
    fi
    
    # 
    if [ ! -f "$TASK_TYPE_FILE" ]; then
        echo "Method,Steps,Task,Seed,CheckpointType,SuccessRate,FLOPs,Speedup" > "$TASK_TYPE_FILE"
    fi
    
    # 
    if [ -f "${task_output_dir}/fast_eval_log.json" ]; then
        # block_pushingkitchen，
        if [ "$task_name" = "block_pushing" ]; then
            # p1p2
            local BP_P1=$(grep "\"test/p1\":" "${task_output_dir}/fast_eval_log.json" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local BP_P2=$(grep "\"test/p2\":" "${task_output_dir}/fast_eval_log.json" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local SPEEDUP=$(grep "speedup" "${task_output_dir}/eval_results.json" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            local FLOPS=$(grep "flops" "${task_output_dir}/eval_results.json" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            
            # 
            echo "BAC,${STEPS},${task_name}_p1,${seed},${checkpoint_type},${BP_P1},${FLOPS},${SPEEDUP}" >> "$RESULTS_FILE"
            echo "BAC,${STEPS},${task_name}_p2,${seed},${checkpoint_type},${BP_P2},${FLOPS},${SPEEDUP}" >> "$RESULTS_FILE"
            
            # 
            echo "BAC,${STEPS},${task_name}_p1,${seed},${checkpoint_type},${BP_P1},${FLOPS},${SPEEDUP}" >> "$BP_RESULTS_FILE"
            echo "BAC,${STEPS},${task_name}_p2,${seed},${checkpoint_type},${BP_P2},${FLOPS},${SPEEDUP}" >> "$BP_RESULTS_FILE"
            
            print_info ": ${task_name} p1=${BP_P1}, p2=${BP_P2} (BAC, : ${STEPS}, : ${seed}, : ${checkpoint_type})"
        elif [ "$task_name" = "kitchen" ]; then
            # p_1p_4
            local KITCHEN_P1=$(grep "\"test/p_1\":" "${task_output_dir}/fast_eval_log.json" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local KITCHEN_P2=$(grep "\"test/p_2\":" "${task_output_dir}/fast_eval_log.json" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local KITCHEN_P3=$(grep "\"test/p_3\":" "${task_output_dir}/fast_eval_log.json" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local KITCHEN_P4=$(grep "\"test/p_4\":" "${task_output_dir}/fast_eval_log.json" | grep -o '[0-9]\+\.[0-9]\+' || echo "0.0")
            local SPEEDUP=$(grep "speedup" "${task_output_dir}/eval_results.json" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            local FLOPS=$(grep "flops" "${task_output_dir}/eval_results.json" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            
            # 
            echo "BAC,${STEPS},${task_name}_p1,${seed},${checkpoint_type},${KITCHEN_P1},${FLOPS},${SPEEDUP}" >> "$RESULTS_FILE"
            echo "BAC,${STEPS},${task_name}_p2,${seed},${checkpoint_type},${KITCHEN_P2},${FLOPS},${SPEEDUP}" >> "$RESULTS_FILE"
            echo "BAC,${STEPS},${task_name}_p3,${seed},${checkpoint_type},${KITCHEN_P3},${FLOPS},${SPEEDUP}" >> "$RESULTS_FILE"
            echo "BAC,${STEPS},${task_name}_p4,${seed},${checkpoint_type},${KITCHEN_P4},${FLOPS},${SPEEDUP}" >> "$RESULTS_FILE"
            
            # 
            echo "BAC,${STEPS},${task_name}_p1,${seed},${checkpoint_type},${KITCHEN_P1},${FLOPS},${SPEEDUP}" >> "$KITCHEN_RESULTS_FILE"
            echo "BAC,${STEPS},${task_name}_p2,${seed},${checkpoint_type},${KITCHEN_P2},${FLOPS},${SPEEDUP}" >> "$KITCHEN_RESULTS_FILE"
            echo "BAC,${STEPS},${task_name}_p3,${seed},${checkpoint_type},${KITCHEN_P3},${FLOPS},${SPEEDUP}" >> "$KITCHEN_RESULTS_FILE"
            echo "BAC,${STEPS},${task_name}_p4,${seed},${checkpoint_type},${KITCHEN_P4},${FLOPS},${SPEEDUP}" >> "$KITCHEN_RESULTS_FILE"
            
            print_info ": ${task_name} p1=${KITCHEN_P1}, p2=${KITCHEN_P2}, p3=${KITCHEN_P3}, p4=${KITCHEN_P4} (BAC, : ${STEPS}, : ${seed}, : ${checkpoint_type})"
        else
            # 
            local SUCCESS_RATE=$(grep "mean_score" "${task_output_dir}/eval_results.json" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "0.0")
            local SPEEDUP=$(grep "speedup" "${task_output_dir}/eval_results.json" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            local FLOPS=$(grep "flops" "${task_output_dir}/eval_results.json" | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "-")
            
            # 
            echo "BAC,${STEPS},${task_name},${seed},${checkpoint_type},${SUCCESS_RATE},${FLOPS},${SPEEDUP}" >> "$RESULTS_FILE"
            
            # 
            if [[ " ${PH_TASKS[*]} " =~ " ${task_name} " ]]; then
                echo "BAC,${STEPS},${task_name},${seed},${checkpoint_type},${SUCCESS_RATE},${FLOPS},${SPEEDUP}" >> "$PH_RESULTS_FILE"
            elif [[ " ${MH_TASKS[*]} " =~ " ${task_name} " ]]; then
                echo "BAC,${STEPS},${task_name},${seed},${checkpoint_type},${SUCCESS_RATE},${FLOPS},${SPEEDUP}" >> "$MH_RESULTS_FILE"
            fi
            
            print_info ": ${task_name} (BAC, : ${STEPS}, : ${seed}, : ${checkpoint_type}, : ${SUCCESS_RATE})"
        fi
    else
        print_warning "，: ${task_output_dir}/fast_eval_log.json"
    fi
}

# 
print_title "BAC"
print_info ": $OUTPUT_DIR"
print_info ": $RESULTS_FILE"

# CHECKPOINT_TYPE
declare -a CHECKPOINT_TYPES=()
if [ "$CHECKPOINT_TYPE" = "all" ]; then
    CHECKPOINT_TYPES=("max" "avg")
else
    CHECKPOINT_TYPES=("$CHECKPOINT_TYPE")
fi

# 
IFS=',' read -ra SEEDS_ARRAY <<< "$SEEDS"
for task_name in "${ACTIVE_TASKS[@]}"; do
    print_subtitle ": $task_name"
    for seed in "${SEEDS_ARRAY[@]}"; do
        print_info ": $seed"
        # 
        for current_checkpoint_type in "${CHECKPOINT_TYPES[@]}"; do
            print_info ": $current_checkpoint_type"
            execute_single_evaluation "$task_name" "$seed" "$current_checkpoint_type"
        done
    done
done

print_title "BAC"
print_info ":"
print_info "- : $RESULTS_FILE"

# 
SUMMARY_FILE="${RESULTS_DIR}/summary.csv"
echo "TaskType,Method,Steps,Task,MaxSuccessRate,AvgSuccessRate,FLOPs,Speedup" > "$SUMMARY_FILE"

# Python
python3 - <<EOF
import pandas as pd
import numpy as np
import os
import glob
import json

# RESULTS_DIR
result_dir = "${RESULTS_DIR}"
print(f": {result_dir}")

# 
all_results = []

# PH
ph_file = os.path.join(result_dir, "ph_results.csv")
if os.path.exists(ph_file):
    try:
        ph_df = pd.read_csv(ph_file)
        ph_df['TaskType'] = 'ph'
        all_results.append(ph_df)
        print(f"PH: {ph_file}")
    except Exception as e:
        print(f"PH: {e}")

# MH
mh_file = os.path.join(result_dir, "mh_results.csv")
if os.path.exists(mh_file):
    try:
        mh_df = pd.read_csv(mh_file)
        mh_df['TaskType'] = 'mh'
        all_results.append(mh_df)
        print(f"MH: {mh_file}")
    except Exception as e:
        print(f"MH: {e}")

# BP
bp_file = os.path.join(result_dir, "bp_results.csv")
if os.path.exists(bp_file):
    try:
        bp_df = pd.read_csv(bp_file)
        bp_df['TaskType'] = 'bp'
        all_results.append(bp_df)
        print(f"BP: {bp_file}")
    except Exception as e:
        print(f"BP: {e}")

# Kitchen
kitchen_file = os.path.join(result_dir, "kitchen_results.csv")
if os.path.exists(kitchen_file):
    try:
        kitchen_df = pd.read_csv(kitchen_file)
        kitchen_df['TaskType'] = 'kitchen'
        all_results.append(kitchen_df)
        print(f"Kitchen: {kitchen_file}")
    except Exception as e:
        print(f"Kitchen: {e}")

# CSV，
if not all_results:
    print("，...")
    
    # 
    tasks = {
        'ph': ['lift_ph', 'can_ph', 'square_ph', 'transport_ph', 'tool_hang_ph', 'pusht'],
        'mh': ['lift_mh', 'can_mh', 'square_mh', 'transport_mh'],
        'bp': ['block_pushing'],
        'kitchen': ['kitchen']
    }
    
    # 
    manual_results = []
    
    # 
    for task_type, task_list in tasks.items():
        for task in task_list:
            task_dir = os.path.join(result_dir, task)
            
            if not os.path.exists(task_dir):
                continue
                
            # 
            for seed_dir in glob.glob(os.path.join(task_dir, "seed*")):
                # 
                dir_name = os.path.basename(seed_dir)
                parts = dir_name.split('_')
                if len(parts) < 2:
                    continue
                    
                seed = parts[0].replace('seed', '')
                checkpoint_type = parts[1]
                
                # 
                result_file = os.path.join(seed_dir, "fast_eval_log.json")
                metrics_file = os.path.join(seed_dir, "eval_results.json")
                
                if not os.path.exists(result_file):
                    continue
                
                # 
                try:
                    # block_pushing，p1p2
                    if task == 'block_pushing':
                        # JSON
                        try:
                            with open(result_file, 'r') as f:
                                result_data = json.load(f)
                                bp_p1 = result_data.get('test/p1', 0.0)
                                bp_p2 = result_data.get('test/p2', 0.0)
                        except:
                            # JSON，grep
                            with open(result_file, 'r') as f:
                                content = f.read()
                                import re
                                bp_p1_match = re.search(r'"test/p1":\s*([\d\.]+)', content)
                                bp_p2_match = re.search(r'"test/p2":\s*([\d\.]+)', content)
                                bp_p1 = float(bp_p1_match.group(1)) if bp_p1_match else 0.0
                                bp_p2 = float(bp_p2_match.group(1)) if bp_p2_match else 0.0
                        
                        # speedupflops
                        speedup = "-"
                        flops = "-"
                        if os.path.exists(metrics_file):
                            try:
                                with open(metrics_file, 'r') as f:
                                    metrics_data = json.load(f)
                                    speedup = metrics_data.get('speedup', '-')
                                    flops = metrics_data.get('flops', '-')
                            except:
                                with open(metrics_file, 'r') as f:
                                    content = f.read()
                                    speedup_match = re.search(r'"speedup":\s*([\d\.]+)', content)
                                    flops_match = re.search(r'"flops":\s*([\d\.]+)', content)
                                    speedup = float(speedup_match.group(1)) if speedup_match else "-"
                                    flops = float(flops_match.group(1)) if flops_match else "-"
                        
                        # p1p2
                        manual_results.append({
                            'Method': 'BAC',
                            'Steps': ${STEPS},
                            'Task': f'{task}_p1',
                            'Seed': seed,
                            'CheckpointType': checkpoint_type,
                            'SuccessRate': bp_p1,
                            'FLOPs': flops,
                            'Speedup': speedup,
                            'TaskType': task_type
                        })
                        
                        manual_results.append({
                            'Method': 'BAC',
                            'Steps': ${STEPS},
                            'Task': f'{task}_p2',
                            'Seed': seed,
                            'CheckpointType': checkpoint_type,
                            'SuccessRate': bp_p2,
                            'FLOPs': flops,
                            'Speedup': speedup,
                            'TaskType': task_type
                        })
                        
                    elif task == 'kitchen':
                        # JSON
                        try:
                            with open(result_file, 'r') as f:
                                result_data = json.load(f)
                                kitchen_p1 = result_data.get('test/p_1', 0.0)
                                kitchen_p2 = result_data.get('test/p_2', 0.0)
                                kitchen_p3 = result_data.get('test/p_3', 0.0)
                                kitchen_p4 = result_data.get('test/p_4', 0.0)
                        except:
                            # JSON，grep
                            with open(result_file, 'r') as f:
                                content = f.read()
                                import re
                                p1_match = re.search(r'"test/p_1":\s*([\d\.]+)', content)
                                p2_match = re.search(r'"test/p_2":\s*([\d\.]+)', content)
                                p3_match = re.search(r'"test/p_3":\s*([\d\.]+)', content)
                                p4_match = re.search(r'"test/p_4":\s*([\d\.]+)', content)
                                kitchen_p1 = float(p1_match.group(1)) if p1_match else 0.0
                                kitchen_p2 = float(p2_match.group(1)) if p2_match else 0.0
                                kitchen_p3 = float(p3_match.group(1)) if p3_match else 0.0
                                kitchen_p4 = float(p4_match.group(1)) if p4_match else 0.0
                        
                        # speedupflops
                        speedup = "-"
                        flops = "-"
                        if os.path.exists(metrics_file):
                            try:
                                with open(metrics_file, 'r') as f:
                                    metrics_data = json.load(f)
                                    speedup = metrics_data.get('speedup', '-')
                                    flops = metrics_data.get('flops', '-')
                            except:
                                with open(metrics_file, 'r') as f:
                                    content = f.read()
                                    speedup_match = re.search(r'"speedup":\s*([\d\.]+)', content)
                                    flops_match = re.search(r'"flops":\s*([\d\.]+)', content)
                                    speedup = float(speedup_match.group(1)) if speedup_match else "-"
                                    flops = float(flops_match.group(1)) if flops_match else "-"
                        
                        # p1-p4
                        for idx, p_value in enumerate([kitchen_p1, kitchen_p2, kitchen_p3, kitchen_p4], 1):
                            manual_results.append({
                                'Method': 'BAC',
                                'Steps': ${STEPS},
                                'Task': f'{task}_p{idx}',
                                'Seed': seed,
                                'CheckpointType': checkpoint_type,
                                'SuccessRate': p_value,
                                'FLOPs': flops,
                                'Speedup': speedup,
                                'TaskType': task_type
                            })
                    else:
                        # 
                        # metrics
                        success_rate = 0.0
                        if os.path.exists(metrics_file):
                            try:
                                with open(metrics_file, 'r') as f:
                                    metrics_data = json.load(f)
                                    success_rate = metrics_data.get('mean_score', 0.0)
                                    speedup = metrics_data.get('speedup', '-')
                                    flops = metrics_data.get('flops', '-')
                            except:
                                with open(metrics_file, 'r') as f:
                                    content = f.read()
                                    success_match = re.search(r'"mean_score":\s*([\d\.]+)', content)
                                    speedup_match = re.search(r'"speedup":\s*([\d\.]+)', content)
                                    flops_match = re.search(r'"flops":\s*([\d\.]+)', content)
                                    success_rate = float(success_match.group(1)) if success_match else 0.0
                                    speedup = float(speedup_match.group(1)) if speedup_match else "-"
                                    flops = float(flops_match.group(1)) if flops_match else "-"
                        
                        manual_results.append({
                            'Method': 'BAC',
                            'Steps': ${STEPS},
                            'Task': task,
                            'Seed': seed,
                            'CheckpointType': checkpoint_type,
                            'SuccessRate': success_rate,
                            'FLOPs': flops,
                            'Speedup': speedup,
                            'TaskType': task_type
                        })
                except Exception as e:
                    print(f" {task} ( {seed},  {checkpoint_type}) : {e}")
    
    # ，DataFrame
    if manual_results:
        results_df = pd.DataFrame(manual_results)
        print(f" {len(manual_results)} ")
    else:
        print("，")
        exit(1)
else:
    # CSV
    results_df = pd.concat(all_results, ignore_index=True)
    print(f"CSV: {results_df.shape}")

# ()
task_display_names = {
    'lift_ph': 'Lift$_{ph}$', 
    'can_ph': 'Can$_{ph}$', 
    'square_ph': 'Square$_{ph}$',
    'transport_ph': 'Trans$_{ph}$', 
    'tool_hang_ph': 'Tool$_{ph}$', 
    'pusht': 'Push--T',
    'lift_mh': 'Lift$_{mh}$', 
    'can_mh': 'Can$_{mh}$', 
    'square_mh': 'Square$_{mh}$', 
    'transport_mh': 'Trans$_{mh}$',
    'block_pushing_p1': 'BP$_{p1}$', 
    'block_pushing_p2': 'BP$_{p2}$',
    'kitchen_p1': 'Kit$_{p1}$', 
    'kitchen_p2': 'Kit$_{p2}$', 
    'kitchen_p3': 'Kit$_{p3}$', 
    'kitchen_p4': 'Kit$_{p4}$'
}

# 
print("\n:")
print(results_df[['Method', 'Steps', 'Task', 'SuccessRate', 'Speedup', 'CheckpointType']].head(10))

# 
summary_data = []

# DataFrame
print(f"\n: {list(results_df.columns)}")
print(f": {results_df['Method'].unique()}")
print(f": {results_df['Steps'].unique()}")
print(f": {results_df['Task'].unique()}")

# 、、、
grouped = results_df.groupby(['TaskType', 'Method', 'Steps'])
print(f": {len(grouped)}")

for (task_type, method, steps), group in grouped:
    print(f": {task_type}, {method}, {steps}, : {len(group)}")
    
    for task in sorted(group['Task'].unique()):
        # 
        task_data = group[group['Task'] == task]
        
        if task_data.empty:
            print(f"  - : {task}")
            continue
            
        # 
        if 'CheckpointType' in task_data.columns:
            max_success = task_data[task_data['CheckpointType'] == 'max']['SuccessRate'].mean()
            avg_success = task_data[task_data['CheckpointType'] == 'avg']['SuccessRate'].mean()
        else:
            # CheckpointType，SuccessRate
            print(f"  - : {task} CheckpointType，")
            max_success = avg_success = task_data['SuccessRate'].mean()
        
        # FLOPs
        if 'FLOPs' in task_data.columns:
            flops = task_data['FLOPs'].mean() if not task_data['FLOPs'].isnull().all() else float('nan')
        else:
            flops = float('nan')
            
        if 'Speedup' in task_data.columns:
            speedup = task_data['Speedup'].mean() if not task_data['Speedup'].isnull().all() else float('nan')
        else:
            speedup = float('nan')
        
        # 
        summary_data.append({
            'TaskType': task_type,
            'Method': method,
            'Steps': steps,
            'Task': task,
            'MaxSuccessRate': max_success,
            'AvgSuccessRate': avg_success,
            'FLOPs': flops,
            'Speedup': speedup
        })
        print(f"  - : {task}, : {max_success:.3f}/{avg_success:.3f}")

# DataFrame
summary_df = pd.DataFrame(summary_data)
print(f": {summary_df.shape}")

# "-"
summary_df = summary_df.fillna('-')

# ，
print("\n\033[1;34m=====  =====\033[0m")

# Python
ph_tasks = ['lift_ph', 'can_ph', 'square_ph', 'transport_ph', 'tool_hang_ph', 'pusht']
mh_tasks = ['lift_mh', 'can_mh', 'square_mh', 'transport_mh']
bp_tasks = ['block_pushing_p1', 'block_pushing_p2']
kitchen_tasks = ['kitchen_p1', 'kitchen_p2', 'kitchen_p3', 'kitchen_p4']
lowdim_tasks = bp_tasks + kitchen_tasks

# PH
print("\n\033[1;36mPH (BAC):\033[0m")
for steps in sorted(summary_df['Steps'].unique()):
    # PH
    ph_task_data = {}
    for task in ph_tasks:
        task_rows = summary_df[(summary_df['Task'] == task) & (summary_df['Steps'] == steps)]
        if not task_rows.empty:
            max_sr = task_rows['MaxSuccessRate'].iloc[0]
            avg_sr = task_rows['AvgSuccessRate'].iloc[0]
            
            # "nan"
            if max_sr != '-' and not pd.isna(max_sr):
                max_sr_str = f"{float(max_sr):.2f}"
            else:
                max_sr_str = "--"
                
            if avg_sr != '-' and not pd.isna(avg_sr):
                avg_sr_str = f"{float(avg_sr):.2f}"
            else:
                avg_sr_str = "--"
                
            ph_task_data[task] = f"{max_sr_str}/{avg_sr_str}"
        else:
            ph_task_data[task] = "--/--"
    
    # ，
    if not ph_task_data:
        continue
    
    # 
    speedup_values = []
    for task in ph_tasks:
        task_rows = summary_df[(summary_df['Task'] == task) & (summary_df['Steps'] == steps)]
        if not task_rows.empty:
            speedup = task_rows['Speedup'].iloc[0]
            if speedup != '-' and not pd.isna(speedup):
                try:
                    speedup_values.append(float(speedup))
                except:
                    pass
    
    # 
    if speedup_values:
        avg_speedup = sum(speedup_values) / len(speedup_values)
        speedup_str = f"{avg_speedup:.2f}"
    else:
        speedup_str = "--"
    
    # 
    ph_row = f"BAC & {steps} & {ph_task_data.get('lift_ph', '--/--')} & {ph_task_data.get('can_ph', '--/--')} & {ph_task_data.get('square_ph', '--/--')} & {ph_task_data.get('transport_ph', '--/--')} & {ph_task_data.get('tool_hang_ph', '--/--')} & {ph_task_data.get('pusht', '--/--')} & -- & {speedup_str}"
    print(ph_row)

# MH
print("\n\033[1;36mMH (BAC):\033[0m")
for steps in sorted(summary_df['Steps'].unique()):
    # MH
    mh_task_data = {}
    for task in mh_tasks:
        task_rows = summary_df[(summary_df['Task'] == task) & (summary_df['Steps'] == steps)]
        if not task_rows.empty:
            max_sr = task_rows['MaxSuccessRate'].iloc[0]
            avg_sr = task_rows['AvgSuccessRate'].iloc[0]
            
            # "nan"
            if max_sr != '-' and not pd.isna(max_sr):
                max_sr_str = f"{float(max_sr):.2f}"
            else:
                max_sr_str = "--"
                
            if avg_sr != '-' and not pd.isna(avg_sr):
                avg_sr_str = f"{float(avg_sr):.2f}"
            else:
                avg_sr_str = "--"
                
            mh_task_data[task] = f"{max_sr_str}/{avg_sr_str}"
        else:
            mh_task_data[task] = "--/--"
    
    # ，
    if not mh_task_data:
        continue
    
    # 
    speedup_values = []
    for task in mh_tasks:
        task_rows = summary_df[(summary_df['Task'] == task) & (summary_df['Steps'] == steps)]
        if not task_rows.empty:
            speedup = task_rows['Speedup'].iloc[0]
            if speedup != '-' and not pd.isna(speedup):
                try:
                    speedup_values.append(float(speedup))
                except:
                    pass
    
    # 
    if speedup_values:
        avg_speedup = sum(speedup_values) / len(speedup_values)
        speedup_str = f"{avg_speedup:.2f}"
    else:
        speedup_str = "--"
    
    # 
    mh_row = f"BAC & {steps} & {mh_task_data.get('lift_mh', '--/--')} & {mh_task_data.get('can_mh', '--/--')} & {mh_task_data.get('square_mh', '--/--')} & {mh_task_data.get('transport_mh', '--/--')} & -- & {speedup_str}"
    print(mh_row)

# LOWDIM
print("\n\033[1;36mLOWDIM (BAC):\033[0m")
for steps in sorted(summary_df['Steps'].unique()):
    # LOWDIM
    lowdim_task_data = {}
    for task in lowdim_tasks:
        task_rows = summary_df[(summary_df['Task'] == task) & (summary_df['Steps'] == steps)]
        if not task_rows.empty:
            max_sr = task_rows['MaxSuccessRate'].iloc[0]
            avg_sr = task_rows['AvgSuccessRate'].iloc[0]
            
            # "nan"
            if max_sr != '-' and not pd.isna(max_sr):
                max_sr_str = f"{float(max_sr):.2f}"
            else:
                max_sr_str = "--"
                
            if avg_sr != '-' and not pd.isna(avg_sr):
                avg_sr_str = f"{float(avg_sr):.2f}"
            else:
                avg_sr_str = "--"
                
            lowdim_task_data[task] = f"{max_sr_str}/{avg_sr_str}"
        else:
            lowdim_task_data[task] = "--/--"
    
    # ，
    if not lowdim_task_data:
        continue
    
    # 
    speedup_values = []
    for task in lowdim_tasks:
        task_rows = summary_df[(summary_df['Task'] == task) & (summary_df['Steps'] == steps)]
        if not task_rows.empty:
            speedup = task_rows['Speedup'].iloc[0]
            if speedup != '-' and not pd.isna(speedup):
                try:
                    speedup_values.append(float(speedup))
                except:
                    pass
    
    # 
    if speedup_values:
        avg_speedup = sum(speedup_values) / len(speedup_values)
        speedup_str = f"{avg_speedup:.2f}"
    else:
        speedup_str = "--"
    
    #  - lowdim
    lowdim_row = f"BAC & {steps} & {lowdim_task_data.get('block_pushing_p1', '--/--')} & {lowdim_task_data.get('block_pushing_p2', '--/--')} & {lowdim_task_data.get('kitchen_p1', '--/--')} & {lowdim_task_data.get('kitchen_p2', '--/--')} & {lowdim_task_data.get('kitchen_p3', '--/--')} & {lowdim_task_data.get('kitchen_p4', '--/--')} & -- & {speedup_str}"
    print(lowdim_row)

# 
summary_file = os.path.join(result_dir, "summary.csv")
summary_df.to_csv(summary_file, index=False)
print(f"\n: {summary_file}")
print(f": {summary_df.shape}")
EOF

print_title "BAC"
print_info ":"
print_info "- : $RESULTS_FILE"
print_info "- : $SUMMARY_FILE" 