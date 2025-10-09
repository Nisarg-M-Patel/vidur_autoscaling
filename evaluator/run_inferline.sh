#!/bin/bash

# Update current directory
cd "$(dirname "$0")"
cd ..

trace_file="data/generated_traces/Synthetic_Trace1_1hr.csv"

# Remove old output files
rm -rf simulator_output/*

# Run the simulator
python3 -m vidur.main \
    --metrics_config_output_dir simulator_output/1 \
    --autoscaler_config_type inferline \
    --request_generator_config_type trace_replay \
    --trace_request_generator_config_trace_file $trace_file \
    --replica_scheduler_config_type vllm \
    --vllm_scheduler_config_batch_size_cap 16 \
    --vllm_scheduler_config_max_tokens_in_batch 4096 \