import xml.etree.ElementTree as ET

def summarize_tripinfo(tripinfo_path):
    """Avg waiting time, avg travel time, throughput, avg stops — from
    SUMO's own tripinfo-output, computed only over vehicles that
    actually completed their trip."""
    tree = ET.parse(tripinfo_path)
    root = tree.getroot()

    waiting_times, durations, stop_counts = [], [], []
    for trip in root.findall("tripinfo"):
        waiting_times.append(float(trip.get("waitingTime")))
        durations.append(float(trip.get("duration")))
        stop_counts.append(int(trip.get("waitingCount")))

    n = len(durations)
    if n == 0:
        return 0, 0, 0, 0

    avg_wait = sum(waiting_times) / n
    avg_travel = sum(durations) / n
    throughput = n
    avg_stops = sum(stop_counts) / n
    return avg_wait, avg_travel, throughput, avg_stops


def summarize_summary_output(summary_path):
    """Avg queue length and avg speed — from SUMO's summary-output,
    averaged across every simulated timestep."""
    tree = ET.parse(summary_path)
    root = tree.getroot()

    halting, speeds = [], []
    for step in root.findall("step"):
        halting.append(float(step.get("halting", 0)))
        speeds.append(float(step.get("meanSpeed", 0)))

    n = len(halting)
    avg_queue = sum(halting) / n if n else 0
    avg_speed = sum(speeds) / n if n else 0
    return avg_queue, avg_speed