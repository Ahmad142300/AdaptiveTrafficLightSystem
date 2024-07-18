import numpy as np
import json


def adjust_for_lanes(car_estimates, lanes):
    """Adjust car estimates based on the number of lanes"""
    adjusted_estimates = []
    for i in range(len(car_estimates)):
        if car_estimates[i] == 0:
            adjusted_estimates.append(0)
        else:
            adjusted_estimates.append(car_estimates[i] / lanes[i])
    return adjusted_estimates


def initial_allocation(adjusted_estimates, cycle_time):
    """Allocate initial green times based on adjusted estimates."""
    green_times = []
    total_adjusted_estimates = sum(adjusted_estimates)
    for estimate in adjusted_estimates:
        if estimate == 0:
            green_times.append(0)
        else:
            green_times.append(int((estimate / total_adjusted_estimates) * cycle_time))
    return green_times


def enforce_green_times(green_times, min_green_time, max_green_time):
    """Enforce minimum and maximum green times for each signal."""
    enforced_times = []
    for time in green_times:
        if time == 0:
            enforced_times.append(0)
        else:
            enforced_times.append(max(min_green_time, min(max_green_time, time)))
    return enforced_times


def adjust_to_cycle_time(green_times, cycle_time):
    """Adjust green times to ensure the total matches the cycle time."""
    total_allocated_time = sum(green_times)
    scale_factor = cycle_time / total_allocated_time if total_allocated_time > 0 else 0
    adjusted_times = []
    for time in green_times:
        adjusted_times.append(int(time * scale_factor))

    # Ensure the total matches the cycle time
    total_allocated_time = sum(adjusted_times)
    if total_allocated_time != cycle_time:
        adjusted_times[-1] += cycle_time - total_allocated_time

    return adjusted_times


def calculate_optimal_green_times(car_estimates, lanes, cycle_time, min_green_time, max_green_time):
    """Calculate the optimal green times for each traffic signal."""
    adjusted_estimates = adjust_for_lanes(car_estimates, lanes)
    green_times = []
    for estimate in adjusted_estimates:
        n = estimate  # Number of vehicles in a lane
        green_time = 3 + 2 * n
        green_times.append(green_time)

    green_times = enforce_green_times(green_times, min_green_time, max_green_time)
    green_times = adjust_to_cycle_time(green_times, cycle_time)
    return green_times


# Load JSON data from file
with open('database.json', 'r') as file:
    data = json.load(file)

# Process each cycle in the data
cycles = {}
for entry in data:
    for cycle, details in entry.items():
        if cycle not in cycles:
            cycles[cycle] = []
        cycles[cycle].append(details)

total_cycle_time = 300  # Total cycle time for all signals (5 minutes)
num_signals = 4  # Number of signals at each intersection
min_green_time = 10  # Minimum green time
max_green_time = 120  # Maximum green time
lanes = [2, 3, 2, 4]  # Example lane counts for each signal

optimal_green_times = {}

for cycle, signals in cycles.items():
    car_estimates = [signal['num_cars'] for signal in signals]
    green_times = calculate_optimal_green_times(car_estimates, lanes, total_cycle_time, min_green_time, max_green_time)
    optimal_green_times[cycle] = []
    for i, signal in enumerate(signals):
        optimal_green_times[cycle].append({
            'trafficID': signal['trafficID'],
            'green_duration': green_times[i]
        })

# Print the results
for cycle, durations in optimal_green_times.items():
    print(f"{cycle}:")
    for duration in durations:
        print(f"  Signal {duration['trafficID']} green for {duration['green_duration']} seconds")
