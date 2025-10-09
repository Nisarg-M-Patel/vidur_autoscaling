import os
import pandas as pd
import matplotlib.pyplot as plt
import json

def get_cost_metrics(output_dir) -> dict:
    """
    Load cost metrics from the output directory
    """
    try:
        cost_per_hour = {}
        file_path = os.path.join(output_dir, "cost_per_hour.json")
        with open(file_path) as f:
            cost_per_hour = json.load(f)
        return cost_per_hour
    except:
        return None

def get_request_attainment_metrics(output_dir):
    """
    Load request attainment metrics from the output directory
    """
    try:
        file_path = os.path.join(output_dir, "request_e2e_time.csv")
        request_e2e_time = pd.read_csv(file_path)
        return request_e2e_time
    except:
        return None
    
def get_percentile_request_e2e_time(request_e2e_time: pd.DataFrame, percentile: int) -> float:
    """
    Get the request end-to-end latency for the provided percentile
    """
    percentile_value = request_e2e_time["cdf"].searchsorted(percentile/100, side="left")
    return request_e2e_time["request_e2e_time"][percentile_value]

def get_percentiles_request_e2e_time(request_e2e_time: pd.DataFrame, percentiles: list) -> dict:
    """
    Get the request end-to-end latency for the provided percentiles
    """
    percentiles_request_e2e_time = {}
    for percentile in percentiles:
        percentiles_request_e2e_time[percentile] = get_percentile_request_e2e_time(request_e2e_time, percentile)
    return percentiles_request_e2e_time

def hypervolume(table_data, reference_cost, reference_time):
    """
    Calculate the normalized hypervolume for the provided table data wrt the reference cost and time
    """
    sorted_table_data = sorted(table_data, key=lambda x: x[1])

    total_cost = [row[1] for row in sorted_table_data]
    e2e_time = [row[3] for row in sorted_table_data] # 90th percentile E2E time

    sorted_indices = sorted(range(len(total_cost)), key=lambda i: total_cost[i])

    # Normalize the cost and time. Anything above the reference cost and time is clipped to the reference cost and time
    normalized_cost = [min(cost, reference_cost) / reference_cost for cost in total_cost]
    normalized_time = [min(time, reference_time) / reference_time for time in e2e_time]

    # Calculate the hypervolume
    hypervolume = (1 - normalized_cost[sorted_indices[-1]])*(1 - normalized_time[sorted_indices[-1]])
    for i in range(len(sorted_indices) - 1):
        hypervolume += (normalized_cost[sorted_indices[i+1]] - normalized_cost[sorted_indices[i]]) * (1 - normalized_time[sorted_indices[i]])

    return hypervolume