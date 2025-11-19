#!/bin/bash

# Get the project root directory (where this script is located)
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${PROJECT_ROOT}"

# Set PYTHONPATH to include the project root so 'models' and 'training' can be imported by the agent
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# =============================================================================
# CONFIGURATION - PLEASE MODIFY THESE PATHS
# =============================================================================

# Path to NavSim OpenScene Logs (containing .pkl files)
# e.g. /data/navsim/openscene-v1.1/navsim_logs/
NAVSIM_LOG_PATH="/data/navsim/navsim_logs"

# Path to Sensor Blobs (containing .jpg, .pcd files)
# e.g. /data/navsim/openscene-v1.1/sensor_blobs/
SENSOR_BLOBS_PATH="/data/navsim/sensor_blobs"

# Path to Metric Cache (optional but recommended for speed)
# If you don't have it, you might need to generate it or point to an empty dir (check navsim docs)
# e.g. /data/navsim/metric_cache/
METRIC_CACHE_PATH="/data/navsim/metric_cache"

# Output Directory for Evaluation Results
OUTPUT_DIR="work_dirs/navsim_eval_mmada"

# MMaDA Model Checkpoint Path
MODEL_PATH="Gen-Verse/MMaDA-8B-MixCoT"

# VQGAN Model Path
VQ_MODEL_PATH="multimodalart/MAGVIT2"

# =============================================================================
# RUN EVALUATION
# =============================================================================

echo "Starting NavSim Evaluation for MMaDA..."
echo "Project Root: ${PROJECT_ROOT}"
echo "Model Path: ${MODEL_PATH}"

# Note: 
# - agent=mmada_agent selects the config we added in config/common/agent/mmada_agent.yaml
# - worker.threads controls parallel workers (process-based)
# - train_test_split=navtest selects the validation split (default in many configs)

python navsim/navsim/planning/script/run_pdm_score.py \
    agent=mmada_agent \
    agent.model_path="${MODEL_PATH}" \
    agent.vq_model_path="${VQ_MODEL_PATH}" \
    navsim_log_path="${NAVSIM_LOG_PATH}" \
    sensor_blobs_path="${SENSOR_BLOBS_PATH}" \
    metric_cache_path="${METRIC_CACHE_PATH}" \
    output_dir="${OUTPUT_DIR}" \
    experiment.name="mmada_eval_run" \
    worker.threads=1 # Reduce threads if OOM issues occur with large models

