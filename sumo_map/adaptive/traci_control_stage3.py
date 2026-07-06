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
NUM_EPISODES = 50

MIN_PHASE_TIME = 10
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

ACTIONS = [0, 1, 2]

# 🔥 One agent per junction
agents = {jid: QAgent(ACTIONS) for jid in JUNCTION_IDS}

# -------------------------------------------------------
# STATE FUNCTIONS
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


def get_state(junction_id):
    queue = get_queue_length(junction_id)
    wait = get_avg_waiting_time(junction_id)

    queue_bin = min(queue // 5, 10)
    wait_bin = min(int(wait // 5), 10)

    return (queue_bin, wait_bin)

# -------------------------------------------------------
# ACTION
# -------------------------------------------------------

last_switch_time = {jid: 0 for jid in JUNCTION_IDS}

def apply_action(junction_id, action, now):
    current_phase = traci.trafficlight.getPhase(junction_id)

    num_phases = len(
        traci.trafficlight.getAllProgramLogics(junction_id)[0].phases
    )

    if action == 0:
        return

    elif action == 1:
        remaining = traci.trafficlight.getNextSwitch(junction_id) - now
        extension = min(5, MAX_EXTENSION)

        traci.trafficlight.setPhaseDuration(
            junction_id,
            remaining + extension
        )

    elif action == 2:
        if now - last_switch_time[junction_id] >= MIN_PHASE_TIME:
            traci.trafficlight.setPhase(
                junction_id,
                (current_phase + 1) % num_phases
            )
            last_switch_time[junction_id] = now

# -------------------------------------------------------
# REWARD
# -------------------------------------------------------

def compute_reward(junction_id):
    queue = get_queue_length(junction_id)
    wait = get_avg_waiting_time(junction_id)
    return -(queue + wait)

# -------------------------------------------------------
# FUEL FUNCTION (NEW 🔥)
# -------------------------------------------------------

def get_total_fuel():
    total = 0
    for veh_id in traci.vehicle.getIDList():
        total += traci.vehicle.getFuelConsumption(veh_id)
    return total

# -------------------------------------------------------
# CSV INIT
# -------------------------------------------------------

with open(LOG_FILE, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "total_reward", "total_fuel"])

# -------------------------------------------------------
# TRAINING LOOP
# -------------------------------------------------------

episode_rewards = []

for episode in range(NUM_EPISODES):

    print(f"\n===== Episode {episode} =====")

    traci.start([SUMO_BINARY, "-c", CONFIG_FILE])

    total_reward_episode = 0
    total_fuel_episode = 0

    try:
        while traci.simulation.getTime() < MAX_SIM_TIME:

            traci.simulationStep()
            now = traci.simulation.getTime()

            prev_states = {}
            actions_taken = {}

            # ACTION SELECTION
            for jid in JUNCTION_IDS:
                state = get_state(jid)
                prev_states[jid] = state

                action = agents[jid].choose_action(state)
                actions_taken[jid] = action

                apply_action(jid, action, now)

            # LEARNING
            step_reward = 0

            for jid in JUNCTION_IDS:
                next_state = get_state(jid)
                reward = compute_reward(jid)

                agents[jid].update(
                    prev_states[jid],
                    actions_taken[jid],
                    reward,
                    next_state
                )

                step_reward += reward

            # FUEL TRACKING
            step_fuel = get_total_fuel()

            total_reward_episode += step_reward
            total_fuel_episode += step_fuel

            # EPSILON DECAY
            for agent in agents.values():
                if agent.epsilon > agent.epsilon_min:
                    agent.epsilon *= agent.epsilon_decay

            # PRINT (ONCE PER SECOND)
            delta_t = traci.simulation.getDeltaT()

            if int(now) % 20 == 0 and abs(now - int(now)) < 1e-6:
                print(f"[Ep {episode}] t={int(now)}s reward={step_reward:.2f}")

    finally:
        traci.close()

    # STORE RESULTS
    episode_rewards.append(total_reward_episode)

    # PRINT SUMMARY
    print(f"Episode {episode} TOTAL REWARD: {total_reward_episode:.2f}")
    print(f"Episode {episode} TOTAL FUEL: {total_fuel_episode:.2f}")

    # SAVE CSV
    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([episode, total_reward_episode, total_fuel_episode])

print("\nTraining complete!")