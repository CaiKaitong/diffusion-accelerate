# Server configuration
SERVER_IP = "0.0.0.0"
SERVER_PORT = 8006

# Model configuration
CHECKPOINT_PATH = "data/outputs/YYYY.MM.DD/HH.MM.SS_train_diffusion_transformer_hybrid_cogact_robot_7d/checkpoints/epoch=0500-train_loss=0.018.ckpt"
USE_EMA = True

# Inference configuration
DEVICE = "cuda:2"
SCHEDULER_TYPE = "DDIM"
NUM_INFERENCE_STEPS = 10
INFERENCE_FREQ = 10.0

# Image configuration
IMAGE_QUALITY = 85
IMAGE_RESIZE = True
MAX_IMAGE_SIZE = (1920, 1080)

# Communication configuration
SOCKET_TIMEOUT = 5.0
BUFFER_SIZE = 4096
ENCODING = 'utf-8'
MAX_CLIENTS = 1

# Logging configuration
VERBOSE = True
LOG_LEVEL = 'INFO'

# BAC Acceleration Config
ENABLE_BAC = True
CACHE_MODE = 'optimal'
CACHE_THRESHOLD = 5
NUM_CACHES = 5
CACHE_METRIC = 'cosine'
NUM_BU_BLOCKS = 3
OPTIMAL_STEPS_DIR = None
