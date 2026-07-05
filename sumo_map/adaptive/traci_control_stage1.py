"""
ATLAS - Stage 1: TraCI connection + live state reading
Connects to SUMO, steps through the simulation, and prints per-junction
queue length / waiting time so we can confirm the read-side works before
adding any control logic.
"""

import os
import sys

# ---- SUMO_HOME setup (needed so traci can find the SUMO python bindings) ----
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please set the SUMO_HOME environment variable (see earlier setup).")

import traci

# ---- CONFIG ----
SUMO_BINARY = "sumo-gui"   # use "sumo" (no -gui) for faster headless runs later
CONFIG_FILE = "smart5_adaptive.sumocfg"  # run this script from inside adaptive/

# Fill this in once you've pasted me the tlLogic ids - placeholder for now
JUNCTION_IDS = [
    "cluster13437517362_4374680526_4374680531_5346620503_#2more",  # main 5-way
    "cluster1936499662_2629436073_621284871",                       # north
    "cluster6616421101_6616421102",                                 # south
    "cluster1646746616_6430722657_7263291410",                      # west
    "cluster1156127277_5346644809",                                 # east
]


def get_queue_length(junction_id):
    """Sum of halted vehicles across all incoming lanes controlled by this junction's TLS."""
    lanes = traci.trafficlight.getControlledLanes(junction_id)
    total_halted = 0
    for lane in set(lanes):  # set() avoids double-counting lanes listed more than once
        total_halted += traci.lane.getLastStepHaltingNumber(lane)
    return total_halted


def get_avg_waiting_time(junction_id):
    """Average waiting time (s) of vehicles across all incoming lanes."""
    lanes = traci.trafficlight.getControlledLanes(junction_id)
    waits = []
    for lane in set(lanes):
        veh_ids = traci.lane.getLastStepVehicleIDs(lane)
        for v in veh_ids:
            waits.append(traci.vehicle.getWaitingTime(v))
    return sum(waits) / len(waits) if waits else 0.0


def main():
    sumo_cmd = [SUMO_BINARY, "-c", CONFIG_FILE]
    traci.start(sumo_cmd)

    step = 0
    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()

            # Print state every 5 seconds of sim time instead of every single step,
            # otherwise this will scroll way too fast to read
            if step % 50 == 0:  # with step-length 0.1, 50 steps = 5s
                print(f"\n--- t={traci.simulation.getTime():.1f}s ---")
                for jid in JUNCTION_IDS:
                    q = get_queue_length(jid)
                    w = get_avg_waiting_time(jid)
                    print(f"  Junction {jid}: queue={q:3d}  avg_wait={w:6.1f}s")

            step += 1
    finally:
        traci.close()


if __name__ == "__main__":
    main()
