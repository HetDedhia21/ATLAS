"""
ATLAS - Stage 1: TraCI connection + live state reading
"""

import os
import sys
import traci
import csv

LOG_FILE = "stage1_metrics.csv"
MAX_SIM_TIME = 400

# ---- SUMO_HOME setup ----
if "SUMO_HOME" in os.environ:
    SUMO_HOME = os.environ["SUMO_HOME"]
    tools = os.path.join(SUMO_HOME, "tools")
    sys.path.append(tools)
else:
    sys.exit("Please set SUMO_HOME environment variable.")

# ---- PATH SETUP ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

SUMO_BINARY = os.path.join(SUMO_HOME, "bin", "sumo.exe")
CONFIG_FILE = os.path.join(BASE_DIR, "smart5_adaptive.sumocfg")

RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
TRIPINFO_FILE = os.path.join(RESULTS_DIR, "tripinfo_stage1.xml")
SUMMARY_FILE = os.path.join(RESULTS_DIR, "summary_stage1.xml")

sumo_cmd = [
    SUMO_BINARY, "-c", CONFIG_FILE,
    "--tripinfo-output", TRIPINFO_FILE,
    "--summary-output", SUMMARY_FILE,
]

# ---- JUNCTION IDS ----
JUNCTION_IDS = [
    "cluster13437517362_4374680526_4374680531_5346620503_#2more",
    "cluster1936499662_2629436073_621284871",
    "cluster6616421101_6616421102",
    "cluster1646746616_6430722657_7263291410",
    "cluster1156127277_5346644809",
]


def get_queue_length(junction_id):
    lanes = traci.trafficlight.getControlledLanes(junction_id)
    total_halted = 0
    for lane in set(lanes):
        total_halted += traci.lane.getLastStepHaltingNumber(lane)
    return total_halted


def get_avg_waiting_time(junction_id):
    lanes = traci.trafficlight.getControlledLanes(junction_id)
    waits = []
    for lane in set(lanes):
        veh_ids = traci.lane.getLastStepVehicleIDs(lane)
        for v in veh_ids:
            waits.append(traci.vehicle.getWaitingTime(v))
    return sum(waits) / len(waits) if waits else 0.0

def get_total_fuel():
    total = 0.0
    for veh_id in traci.vehicle.getIDList():
        total += traci.vehicle.getFuelConsumption(veh_id)
    return total


def main():
    traci.start(sumo_cmd)

    # ✅ Create CSV file with header
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time",
            "junction_id",
            "queue_length",
            "avg_waiting_time"
        ])

    step = 0
    total_fuel = 0.0

    try:
        while (
            traci.simulation.getMinExpectedNumber() > 0
            and traci.simulation.getTime() < MAX_SIM_TIME
        ):

            traci.simulationStep()

            # ✅ FIX: define time AFTER simulation step
            now = traci.simulation.getTime()
            total_fuel += get_total_fuel()

            with open(LOG_FILE, "a", newline="") as f:
                writer = csv.writer(f)

                for jid in JUNCTION_IDS:

                    queue = get_queue_length(jid)
                    wait = get_avg_waiting_time(jid)

                    writer.writerow([
                        now,
                        jid,
                        queue,
                        round(wait, 2)
                    ])

            step += 1

    finally:
        traci.close()

    from utils.metrics_utils import summarize_tripinfo, summarize_summary_output
    avg_wait, avg_travel, throughput, avg_stops = summarize_tripinfo(TRIPINFO_FILE)
    avg_queue, avg_speed = summarize_summary_output(SUMMARY_FILE)

    with open("stage1_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "avg_waiting_time", "avg_queue_length", "avg_travel_time",
            "throughput", "avg_speed", "num_stops", "fuel_consumption"
        ])
        writer.writerow([
            round(avg_wait, 2), round(avg_queue, 2), round(avg_travel, 2),
            throughput, round(avg_speed, 2), round(avg_stops, 2), round(total_fuel, 2)
        ])
    print("Stage 1 summary written to stage1_summary.csv")

if __name__ == "__main__":
    main()