#!/bin/bash

# Update current directory
cd "$(dirname "$0")"
cd ..

trace_file="data/generated_traces/Synthetic_Trace1_1hr.csv"

# Remove old output files
rm -rf simulator_output/*

# Run the simulator with service_levels = [1, 2, 3]
for service_level in 1 2 3
do
    python -m vidur.main \
    --metrics_config_output_dir simulator_output/$service_level \
    --autoscaler_config_type custom \
    --custom_autoscaler_config_service_level $service_level \
    --request_generator_config_type trace_replay \
    --trace_request_generator_config_trace_file $trace_file \
    --replica_scheduler_config_type vllm \
    --vllm_scheduler_config_batch_size_cap 16 \
    --vllm_scheduler_config_max_tokens_in_batch 4096 \

done