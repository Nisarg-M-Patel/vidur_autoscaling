#!/bin/bash

config_file="vidur/config/config.py"
inferline_autoscaler_file="vidur/autoscaler/inferline_autoscaler.py"
custom_autoscaler_file="vidur/autoscaler/custom_autoscaler.py"
round_robin_global_scheduler_file="vidur/scheduler/global_scheduler/round_robin_global_scheduler.py"
lor_global_scheduler_file="vidur/scheduler/global_scheduler/lor_global_scheduler.py"

zip_file="submission.zip"

for file in "$config_file" "$inferline_autoscaler_file" "$custom_autoscaler_file" "$round_robin_global_scheduler_file" "$lor_global_scheduler_file"; do
    if [ ! -f "$file" ]; then
        echo "Error: File '$file' not found!"
        exit 1
    fi
done

zip -j "$zip_file" "$config_file" "$inferline_autoscaler_file" "$custom_autoscaler_file" "$round_robin_global_scheduler_file" "$lor_global_scheduler_file"

echo "Successfully created $zip_file with the specified files at the root level."
