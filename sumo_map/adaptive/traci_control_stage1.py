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

SUMO_BINARY = os.path.join(SUMO_HOME, "bin", "sumo-gui.exe")
CONFIG_FILE = os.path.join(BASE_DIR, "smart5_adaptive.sumocfg")

sumo_cmd = [SUMO_BINARY, "-c", CONFIG_FILE]

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

    try:
        while (
            traci.simulation.getMinExpectedNumber() > 0
            and traci.simulation.getTime() < MAX_SIM_TIME
        ):

            traci.simulationStep()

            # ✅ FIX: define time AFTER simulation step
            now = traci.simulation.getTime()

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

if __name__ == "__main__":
    main()