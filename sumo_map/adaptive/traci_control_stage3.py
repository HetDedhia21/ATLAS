"""
ATLAS - Stage 3: RL-ready traffic signal control (FIXED VERSION)

Improvements:
- Enforces minimum phase duration
- Caps green extension
- Prevents unrealistic rapid switching
"""

import os
import sys
import traci
import csv
import random

LOG_FILE = "stage3_metrics.csv"
MAX_SIM_TIME = 400

# 🔥 NEW CONSTRAINTS
MIN_PHASE_TIME = 10      # minimum seconds before switching again
MAX_EXTENSION = 10       # max extra green time

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please set the SUMO_HOME environment variable.")

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SUMO_BINARY = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo-gui.exe")
CONFIG_FILE = os.path.join(BASE_DIR, "smart5_adaptive.sumocfg")

JUNCTION_IDS = [
    "cluster13437517362_4374680526_4374680531_5346620503_#2more",
    "cluster1936499662_2629436073_621284871",
    "cluster6616421101_6616421102",
    "cluster1646746616_6430722657_7263291410",
    "cluster1156127277_5346644809",
]

# 🔥 TRACK LAST SWITCH TIME PER JUNCTION
last_switch_time = {jid: 0 for jid in JUNCTION_IDS}


# -------------------------------------------------------
# RL COMPONENTS
# -------------------------------------------------------

def get_queue_length(junction_id):
    lanes = traci.trafficlight.getControlledLanes(junction_id)
    return sum(
        traci.lane.getLastStepHaltingNumber(lane)
        for lane in set(lanes)
    )


def get_avg_waiting_time(junction_id):
    lanes = traci.trafficlight.getControlledLanes(junction_id)

    waits = []
    for lane in set(lanes):
        for veh in traci.lane.getLastStepVehicleIDs(lane):
            waits.append(traci.vehicle.getWaitingTime(veh))

    return sum(waits) / len(waits) if waits else 0.0


# ---------------- STATE ----------------

def get_state(junction_id):
    return [
        get_queue_length(junction_id),
        get_avg_waiting_time(junction_id)
    ]


# ---------------- ACTION ----------------

def apply_action(junction_id, action, now):

    current_phase = traci.trafficlight.getPhase(junction_id)

    num_phases = len(
        traci.trafficlight.getAllProgramLogics(junction_id)[0].phases
    )

    # 0 = do nothing
    if action == 0:
        return

    # 1 = extend green (CAPPED)
    elif action == 1:
        remaining = traci.trafficlight.getNextSwitch(junction_id) - now

        # 🔥 cap extension
        extension = min(5, MAX_EXTENSION)

        traci.trafficlight.setPhaseDuration(
            junction_id,
            remaining + extension
        )

    # 2 = switch phase (WITH MIN TIME CHECK)
    elif action == 2:

        # 🔥 enforce minimum phase time
        if now - last_switch_time[junction_id] >= MIN_PHASE_TIME:

            traci.trafficlight.setPhase(
                junction_id,
                (current_phase + 1) % num_phases
            )

            last_switch_time[junction_id] = now


# ---------------- REWARD ----------------

def compute_reward(junction_id):
    queue = get_queue_length(junction_id)
    wait = get_avg_waiting_time(junction_id)

    return -(queue + wait)


# ---------------- DUMMY AGENT ----------------

class DummyAgent:
    def predict(self, state):
        return random.choice([0, 1, 2])


agent = DummyAgent()

# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():

    sumo_cmd = [
        SUMO_BINARY,
        "-c",
        CONFIG_FILE,
    ]

    traci.start(sumo_cmd)

    # Initialize CSV
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

            now = traci.simulation.getTime()

            total_reward = 0

            # -----------------------------------------
            # RL LOOP
            # -----------------------------------------

            for jid in JUNCTION_IDS:

                state = get_state(jid)

                action = agent.predict(state)

                apply_action(jid, action, now)

                reward = compute_reward(jid)
                total_reward += reward

            # -----------------------------------------
            # CSV LOGGING
            # -----------------------------------------

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

            if step % 50 == 0:
                print(f"t={now:.1f}s | total reward={total_reward:.2f}")

            step += 1

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")

    finally:
        traci.close()


if __name__ == "__main__":
    main()