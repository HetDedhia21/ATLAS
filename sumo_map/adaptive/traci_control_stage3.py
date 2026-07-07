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
MAX_STEPS = 1000   # ✅ NEW (optional step-based cap)
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
    wait = min(get_avg_waiting_time(junction_id), 50)

    queue_bin = min(queue // 5, 10)
    wait_bin = min(int(wait // 5), 10)

    phase = traci.trafficlight.getPhase(junction_id)
    return (queue_bin, wait_bin, phase)

def get_pressure(junction_id):
    lanes = traci.trafficlight.getControlledLanes(junction_id)

    incoming = 0
    outgoing = 0

    for lane in set(lanes):
        incoming += traci.lane.getLastStepVehicleNumber(lane)
        outgoing += traci.lane.getLastStepHaltingNumber(lane) * 0.5

    pressure = incoming - outgoing

    return pressure / 10   # 🔥 normalize (CRITICAL)

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
        extension = min(remaining, MAX_EXTENSION)

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

prev_metrics = {jid: {"queue": 0, "wait": 0} for jid in JUNCTION_IDS}

def compute_reward(junction_id):
    queue = get_queue_length(junction_id)
    wait = get_avg_waiting_time(junction_id)
    pressure = get_pressure(junction_id)

    reward = - (
        queue * 0.6 +
        wait * 0.25 +
        pressure * 0.15
    )

    # small bonus
    if queue < 5:
        reward += 2

    # 🔥 ADD THIS (CRITICAL)
    reward = max(min(reward, 50), -50)

    return reward

# -------------------------------------------------------
# FUEL FUNCTION
# -------------------------------------------------------

def get_total_fuel():
    total = 0
    for veh_id in traci.vehicle.getIDList():
        total += traci.vehicle.getFuelConsumption(veh_id)
    return total

# -------------------------------------------------------
# CSV INIT
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
            "epsilon",
            "epsilon_decay",
            "steps"
        ])

# -------------------------------------------------------
# TRAINING LOOP
# -------------------------------------------------------

episode_rewards = []

for episode in range(NUM_EPISODES):

    total_queue_episode = 0
    total_wait_episode = 0
    step_count = 0
    step = 0  # ✅ NEW step counter

    print(f"\n===== Episode {episode} =====")

    for jid in JUNCTION_IDS:
        prev_metrics[jid] = {"queue": 0, "wait": 0}

    traci.start([
        SUMO_BINARY,
        "-c", CONFIG_FILE,
        "--random"
    ])

    total_reward_episode = 0
    total_fuel_episode = 0
    prev_fuel = get_total_fuel()  # ✅ FIX

    try:
        while (
            traci.simulation.getTime() < MAX_SIM_TIME
            and step < MAX_STEPS   # ✅ NOW YOU HAVE step < max_step
        ):

            traci.simulationStep()
            now = traci.simulation.getTime()
            step += 1

            prev_states = {}
            actions_taken = {}

            # ACTIONS
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

            # METRICS
            for jid in JUNCTION_IDS:
                total_queue_episode += get_queue_length(jid)
                total_wait_episode += get_avg_waiting_time(jid)

            step_count += len(JUNCTION_IDS)

            # FUEL
            current_fuel = get_total_fuel()
            step_fuel = current_fuel - prev_fuel
            prev_fuel = current_fuel

            total_fuel_episode += step_fuel
            total_reward_episode += step_reward

            # ✅ CLEAN PRINT
            if step % 20 == 0:
                print(f"[Ep {episode}] step={step} t={int(now)}s reward={step_reward:.2f}")

        # EPSILON DECAY
        for agent in agents.values():
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay

    finally:
        traci.close()

    avg_queue = total_queue_episode / step_count if step_count else 0
    avg_wait = total_wait_episode / step_count if step_count else 0
    epsilon_value = list(agents.values())[0].epsilon

    episode_rewards.append(total_reward_episode)

    print(f"Episode {episode} TOTAL REWARD: {total_reward_episode:.2f}")
    print(f"Episode {episode} TOTAL FUEL: {total_fuel_episode:.2f}")
    print(f"Avg Queue: {avg_queue:.2f} | Avg Wait: {avg_wait:.2f} | Epsilon: {epsilon_value:.3f}")

    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        epsilon_decay_value = list(agents.values())[0].epsilon_decay
        writer.writerow([
            episode,
            total_reward_episode,
            total_fuel_episode,
            avg_queue,
            avg_wait,
            epsilon_value,
            epsilon_decay_value,
            step_count
        ])

print("\nTraining complete!")