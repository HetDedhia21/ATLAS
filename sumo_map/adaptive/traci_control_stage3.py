# -------------------- SAME IMPORTS --------------------
import os
import sys
import traci
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

sys.path.append(PROJECT_ROOT)

from agents.agent import QAgent

LOG_FILE = "stage3_metrics.csv"
MAX_SIM_TIME = 400
MAX_STEPS = 1000
NUM_EPISODES = 100

MIN_PHASE_TIME = 25
MAX_EXTENSION = 10

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please set the SUMO_HOME environment variable.")

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

SUMO_BINARY = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo.exe")
CONFIG_FILE = os.path.join(BASE_DIR, "smart5_adaptive.sumocfg")

JUNCTION_IDS = [
    "cluster13437517362_4374680526_4374680531_5346620503_#2more",
    "cluster1936499662_2629436073_621284871",
    "cluster6616421101_6616421102",
    "cluster1646746616_6430722657_7263291410",
    "cluster1156127277_5346644809", 
]

# -------------------------------------------------------
# NEIGHBORS
# -------------------------------------------------------

NEIGHBORS = {
    JUNCTION_IDS[0]: [JUNCTION_IDS[1], JUNCTION_IDS[2]],
    JUNCTION_IDS[1]: [JUNCTION_IDS[0], JUNCTION_IDS[3]],
    JUNCTION_IDS[2]: [JUNCTION_IDS[0], JUNCTION_IDS[4]],
    JUNCTION_IDS[3]: [JUNCTION_IDS[1]],
    JUNCTION_IDS[4]: [JUNCTION_IDS[2]],
}

ACTIONS = [0, 1, 2]
agents = {jid: QAgent(ACTIONS) for jid in JUNCTION_IDS}

# -------------------------------------------------------
# STATE FUNCTIONS (UNCHANGED)
# -------------------------------------------------------

def get_queue_length(junction_id):
    lanes = traci.trafficlight.getControlledLanes(junction_id)
    return sum(traci.lane.getLastStepHaltingNumber(lane) for lane in set(lanes))


def get_avg_waiting_time(junction_id):
    lanes = traci.trafficlight.getControlledLanes(junction_id)
    waits = []

    for lane in set(lanes):
        for veh in traci.lane.getLastStepVehicleIDs(lane):
            waits.append(traci.vehicle.getWaitingTime(veh))

    return sum(waits) / len(waits) if waits else 0.0


def get_pressure(junction_id):
    lanes = traci.trafficlight.getControlledLanes(junction_id)
    incoming = sum(traci.lane.getLastStepVehicleNumber(l) for l in set(lanes))
    outgoing = sum(traci.lane.getLastStepHaltingNumber(l) for l in set(lanes)) * 0.5
    return (incoming - outgoing) / 10


def get_state(junction_id):
    queue = get_queue_length(junction_id)
    wait = min(get_avg_waiting_time(junction_id), 50)
    pressure = get_pressure(junction_id)

    queue_bin = min(queue // 10, 4)
    wait_bin = min(int(wait // 15), 4)
    pressure_bin = int(max(min(pressure, 10), -10) // 4)

    phase = traci.trafficlight.getPhase(junction_id)

    return (queue_bin, wait_bin, pressure_bin, phase)

# -------------------------------------------------------
# ACTION (UPDATED WITH MAX TIME)
# -------------------------------------------------------

last_switch_time = {jid: 0 for jid in JUNCTION_IDS}

def apply_action(junction_id, action, now):
    current_phase = traci.trafficlight.getPhase(junction_id)
    num_phases = len(traci.trafficlight.getAllProgramLogics(junction_id)[0].phases)

    if action == 0:
        return

    elif action == 1:
        remaining = traci.trafficlight.getNextSwitch(junction_id) - now
        extension = min(remaining, MAX_EXTENSION)
        traci.trafficlight.setPhaseDuration(junction_id, remaining + extension)

    elif action == 2:
        is_green = (current_phase % 2 == 0)

        time_in_phase = now - last_switch_time[junction_id]

        # ✅ FIXED CONTROL
        if is_green and MIN_PHASE_TIME <= time_in_phase:
            traci.trafficlight.setPhase(junction_id, (current_phase + 1) % num_phases)
            last_switch_time[junction_id] = now

# -------------------------------------------------------
# REWARD (UNCHANGED GOOD VERSION)
# -------------------------------------------------------

prev_metrics = {jid: {"queue": 0, "wait": 0} for jid in JUNCTION_IDS}

def compute_reward(junction_id, new_data, all_new_metrics):
    queue = new_data["queue"]
    wait = new_data["wait"]
    pressure = get_pressure(junction_id)

    prev_q = prev_metrics[junction_id]["queue"]
    prev_w = prev_metrics[junction_id]["wait"]

    delta_q = (prev_q - queue) / (prev_q + 1)
    delta_w = (prev_w - wait) / (prev_w + 1)

    throughput_reward = -0.3 * queue

    neighbor_dq = 0
    neighbors = NEIGHBORS.get(junction_id, [])

    if neighbors:
        for n in neighbors:
            n_queue = all_new_metrics[n]["queue"]
            n_prev_q = prev_metrics[n]["queue"]
            neighbor_dq += (n_prev_q - n_queue) / (n_prev_q + 1)

        neighbor_avg_dq = neighbor_dq / len(neighbors)
    else:
        neighbor_avg_dq = 0

    neighbor_avg_dq *= 2.0

    reward = (
        5.0 * delta_q +
        2.0 * delta_w +
        1.0 * neighbor_avg_dq -
        0.3 * abs(pressure) +
        throughput_reward
    )

    return reward, delta_q, delta_w, pressure, neighbor_avg_dq

# -------------------------------------------------------
# CSV INIT (UPDATED)
# -------------------------------------------------------

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode",
            "total_reward",
            "total_fuel",
            "avg_queue",
            "avg_wait",
            "avg_pressure",
            "avg_delta_q",
            "avg_delta_w",
            "teleports",   # ✅ NEW
            "epsilon"
        ])

# -------------------------------------------------------
# TRAINING LOOP
# -------------------------------------------------------

for episode in range(NUM_EPISODES):

    teleport_count = 0   # ✅ RESET PER EPISODE

    total_queue_episode = 0
    total_wait_episode = 0
    step_count = 0
    step = 0
    total_pressure = 0
    total_delta_q = 0
    total_delta_w = 0

    print(f"\n===== Episode {episode} =====")

    for jid in JUNCTION_IDS:
        prev_metrics[jid] = {"queue": 0, "wait": 0}

    traci.start([SUMO_BINARY, "-c", CONFIG_FILE, "--seed", "42"])

    total_reward_episode = 0

    try:
        while traci.simulation.getTime() < MAX_SIM_TIME and step < MAX_STEPS:

            traci.simulationStep()

            # ✅ TELEPORT TRACKING
            teleport_count += traci.simulation.getStartingTeleportNumber()

            now = traci.simulation.getTime()
            step += 1

            prev_states = {}
            actions_taken = {}

            for jid in JUNCTION_IDS:
                state = get_state(jid)
                prev_states[jid] = state

                action = agents[jid].choose_action(state)
                actions_taken[jid] = action

                apply_action(jid, action, now)

            new_metrics = {}
            step_reward = 0

            for jid in JUNCTION_IDS:
                new_metrics[jid] = {
                    "queue": get_queue_length(jid),
                    "wait": get_avg_waiting_time(jid)
                }

            for jid in JUNCTION_IDS:
                next_state = get_state(jid)

                reward, dq, dw, pressure, ndq = compute_reward(
                    jid,
                    new_metrics[jid],
                    new_metrics
                )

                agents[jid].update(
                    prev_states[jid],
                    actions_taken[jid],
                    reward,
                    next_state
                )

                step_reward += reward
                total_delta_q += dq
                total_delta_w += dw
                total_pressure += pressure

            for jid in JUNCTION_IDS:
                prev_metrics[jid] = new_metrics[jid]
                total_queue_episode += new_metrics[jid]["queue"]
                total_wait_episode += new_metrics[jid]["wait"]

            step_count += 1
            total_reward_episode += step_reward

    finally:
        traci.close()

    avg_queue = total_queue_episode / step_count if step_count else 0
    avg_wait = total_wait_episode / step_count if step_count else 0
    avg_pressure = total_pressure / step_count if step_count else 0
    avg_dq = total_delta_q / step_count if step_count else 0
    avg_dw = total_delta_w / step_count if step_count else 0
    avg_teleports = teleport_count   # ✅ FINAL VALUE

    epsilon_value = list(agents.values())[0].epsilon

    print(f"Teleports: {avg_teleports}")

    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            episode,
            total_reward_episode,
            0,
            avg_queue,
            avg_wait,
            avg_pressure,
            avg_dq,
            avg_dw,
            avg_teleports,   # ✅ SAVED
            epsilon_value
        ])

print("\nTraining complete!")