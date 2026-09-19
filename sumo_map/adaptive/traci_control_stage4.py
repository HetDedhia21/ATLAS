"""
ATLAS - Stage 4: Cooperative QMIX with neighbor communication
(paper Sections III-D through IV-G).

Builds directly on Stage 3 (traci_control_stage3.py) - same junctions,
same SUMO config, same reward formula (eq. 1) - but:
  - agents are neural Q-networks (agents/qnetwork.py) instead of a
    tabular q_table, one per junction, with the input augmented by
    neighbor messages (Sec IV-C)
  - a centralized mixing network (agents/mixing_network.py) combines
    the five agent Q-values into Q_tot during training only
  - transitions are stored in a joint replay buffer (agents/replay_buffer.py)
    and trained with TD loss against a periodically hard-updated target
  - at evaluation/deployment, the mixing network and global state are
    discarded; each agent acts from its own Q-network + neighbor
    messages alone (Sec IV-E)

Unlike Stage 3's NEIGHBORS dict (a tree), this stage uses the paper's
actual star topology (Sec III-D): J0 connects directly to all four
peripheral junctions, with no peripheral-to-peripheral link.
"""

import os
import queue
import sys
import traci
import csv
import torch
import torch.nn.functional as F
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

from agents.qnetwork import QMixAgent
from agents.mixing_network import MixingNetwork
from agents.replay_buffer import ReplayBuffer

LOG_FILE = "stage4_metrics.csv"
JUNCTION_LOG_FILE = "stage4_junction_metrics.csv"
ACTION_LOG_FILE = "stage4_action_distribution.csv"
MAX_SIM_TIME = 400
MAX_STEPS = 1000
NUM_EPISODES = 200
TRAIN_SEED_POOL = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]

MIN_PHASE_TIME = 25
MAX_EXTENSION = 10
MAX_PHASE_DURATION = 50

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
    "cluster13437517362_4374680526_4374680531_5346620503_#2more",  # J0 - central hub
    "cluster1936499662_2629436073_621284871",                       # JN - north
    "cluster6616421101_6616421102",                                 # JS - south
    "cluster1646746616_6430722657_7263291410",                      # JW - west
    "cluster1156127277_5346644809",                                 # JE - east
]
J0 = JUNCTION_IDS[0]
JN_ID = JUNCTION_IDS[1]  # for the JN-specific debug trace below
PERIPHERALS = JUNCTION_IDS[1:]

# short, readable names for logging (order matches JUNCTION_IDS above)
JUNCTION_LABELS = {
    JUNCTION_IDS[0]: "J0_central",
    JUNCTION_IDS[1]: "JN_north",
    JUNCTION_IDS[2]: "JS_south",
    JUNCTION_IDS[3]: "JW_west",
    JUNCTION_IDS[4]: "JE_east",
}

# -------------------------------------------------------
# NEIGHBORS - star topology (paper Sec III-D), NOT Stage 3's tree
# -------------------------------------------------------

NEIGHBORS = {
    J0: list(PERIPHERALS),
    **{jid: [J0] for jid in PERIPHERALS},
}

# lane count per junction, used to normalize pressure/throughput reward
# terms by junction size (filled in lazily on first use, since it
# needs an active traci connection)
_lane_count_cache = {}

def get_lane_count(junction_id):
    if junction_id not in _lane_count_cache:
        lanes = traci.trafficlight.getControlledLanes(junction_id)
        _lane_count_cache[junction_id] = max(1, len(set(lanes)))
    return _lane_count_cache[junction_id]

ACTIONS = [0, 1, 2]  # hold / extend / switch, same encoding as Stage 3
NUM_ACTIONS = len(ACTIONS)

# -------------------------------------------------------
# QMIX HYPERPARAMETERS (paper leaves these "configurable, not fixed" -
# Table VIII / Sec IV-E; standard QMIX-literature defaults used here)
# -------------------------------------------------------

LEARNING_RATE = 5e-4
GAMMA = 0.9                # matches Stage 3's QAgent.gamma for comparability
BATCH_SIZE = 32
BUFFER_CAPACITY = 25000
MIN_BUFFER_BEFORE_TRAIN = 200
TARGET_UPDATE_EVERY_N_STEPS = 200
EMBED_DIM = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------
# OBSERVATION / STATE DIMENSIONS (paper Sec III-F, Table II, Sec III-K)
# -------------------------------------------------------
# local obs per agent: [q, w, q_bin, w_bin, phase]                -> 5 values
# neighbor message per neighbor: [q, w, phase]                    -> 3 values
LOCAL_OBS_DIM = 5
MSG_DIM = 3


def obs_dim_for(jid):
    return LOCAL_OBS_DIM + MSG_DIM * len(NEIGHBORS[jid])


# global state s = concatenation of (q, w, phase) for all 5 junctions,
# fixed JUNCTION_IDS order (Sec III-E)
STATE_DIM = 3 * len(JUNCTION_IDS)

# -------------------------------------------------------
# BUILD AGENTS + MIXING NETWORK + REPLAY BUFFER
# -------------------------------------------------------

agents = {
    jid: QMixAgent(obs_dim_for(jid), NUM_ACTIONS, device=DEVICE)
    for jid in JUNCTION_IDS
}

mixing_network = MixingNetwork(len(JUNCTION_IDS), STATE_DIM, embed_dim=EMBED_DIM).to(DEVICE)
target_mixing_network = MixingNetwork(len(JUNCTION_IDS), STATE_DIM, embed_dim=EMBED_DIM).to(DEVICE)
target_mixing_network.load_state_dict(mixing_network.state_dict())

replay_buffer = ReplayBuffer(capacity=BUFFER_CAPACITY)

all_params = list(mixing_network.parameters())
for agent in agents.values():
    all_params += list(agent.q_network.parameters())
optimizer = torch.optim.Adam(all_params, lr=LEARNING_RATE)

train_step_counter = 0

# -------------------------------------------------------
# STATE FUNCTIONS (same as Stage 3)
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
    return (incoming - outgoing) / (10 * get_lane_count(junction_id))


def get_total_fuel():
    total = 0.0
    for veh_id in traci.vehicle.getIDList():
        total += traci.vehicle.getFuelConsumption(veh_id)
    return total


def get_raw_metrics(junction_id):
    """Undiscretized (q, w, phase) - used for messages, global state, reward."""
    queue = get_queue_length(junction_id)
    wait = min(get_avg_waiting_time(junction_id), 50)
    phase = traci.trafficlight.getPhase(junction_id)
    return {"queue": queue, "wait": wait, "phase": phase}


# -------------------------------------------------------
# OBSERVATION BUILDING (paper Sec III-F Table II, Sec III-K, Sec IV-C)
# -------------------------------------------------------


def local_obs_vector(raw):
    """[q, w, q_bin, w_bin, phase], normalized, per Table II."""
    q_bin = min(raw["queue"] // 5, 9)
    w_bin = min(int(raw["wait"] // 5), 9)
    return [
        raw["queue"] / 50.0,
        raw["wait"] / 50.0,
        q_bin / 10.0,
        w_bin / 10.0,
        raw["phase"] / 10.0,
    ]


def message_vector(raw):
    """m_i->j = (q_i, w_i, phi_i), Sec III-K / Table IV."""
    return [raw["queue"] / 50.0, raw["wait"] / 50.0, raw["phase"] / 10.0]


def build_all_obs(all_raw):
    """
    For each agent, concatenate its local obs with the messages it
    receives from its neighbors (fixed order = NEIGHBORS[jid]).
    J0 receives 4 messages, each peripheral receives 1 (Sec III-K).
    """
    obs = {}
    for jid in JUNCTION_IDS:
        vec = local_obs_vector(all_raw[jid])
        for neighbor in NEIGHBORS[jid]:
            vec += message_vector(all_raw[neighbor])
        obs[jid] = vec
    return obs


def build_global_state(all_raw):
    """s = [(q1,w1,phi1), ..., (q5,w5,phi5)] in JUNCTION_IDS order (Sec III-E)."""
    state = []
    for jid in JUNCTION_IDS:
        state += message_vector(all_raw[jid])
    return state


# -------------------------------------------------------
# ACTION (identical encoding to Stage 3)
# -------------------------------------------------------

last_switch_time = {jid: 0 for jid in JUNCTION_IDS}
phase_start_time = {jid: 0 for jid in JUNCTION_IDS}
last_seen_phase = {jid: -1 for jid in JUNCTION_IDS}

# DEBUG: prints exactly what each action does, and whether it actually
# changed anything in SUMO. Only fires for the first 2 episodes so the
# output stays readable. Set to False once you've diagnosed the issue.
DEBUG_ACTIONS = False
DEBUG_MAX_EPISODES = 0


def apply_action(junction_id, action, now):
    current_phase = traci.trafficlight.getPhase(junction_id)
    num_phases = len(traci.trafficlight.getAllProgramLogics(junction_id)[0].phases)
    if current_phase != last_seen_phase[junction_id]:
        last_seen_phase[junction_id] = current_phase
        phase_start_time[junction_id] = now

    debug_on = DEBUG_ACTIONS and episode < DEBUG_MAX_EPISODES

    if action == 0:
        if debug_on:
            print(f"[t={now:.0f}] {JUNCTION_LABELS[junction_id]} action=HOLD phase={current_phase} (no-op)")
        return
    elif action == 1:
        remaining_before = traci.trafficlight.getNextSwitch(junction_id) - now
        max_allowed_remaining = (phase_start_time[junction_id] + MAX_PHASE_DURATION) - now
        extension = max(0, min(MAX_EXTENSION, max_allowed_remaining - remaining_before))
        if extension > 0:
            traci.trafficlight.setPhaseDuration(junction_id, remaining_before + extension)
        remaining_after = traci.trafficlight.getNextSwitch(junction_id) - now
        if debug_on:
            print(
                f"[t={now:.0f}] {JUNCTION_LABELS[junction_id]} action=EXTEND phase={current_phase} "
                f"remaining_before={remaining_before:.1f} extension={extension:.1f} "
                f"remaining_after={remaining_after:.1f} "
                f"{'CHANGED' if abs(remaining_after - remaining_before) > 0.01 else 'NO EFFECT (capped)'}"
            )

    elif action == 2:
        is_green = (current_phase % 2 == 0)
        time_in_phase = now - last_switch_time[junction_id]
        if is_green and MIN_PHASE_TIME <= time_in_phase:
            traci.trafficlight.setPhase(junction_id, (current_phase + 1) % num_phases)
            last_switch_time[junction_id] = now


# -------------------------------------------------------
# REWARD (eq. 1, unchanged from Stage 3, but over the star NEIGHBORS)
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

    throughput_reward = -0.3 * (queue / get_lane_count(junction_id))

    neighbors = NEIGHBORS.get(junction_id, [])
    if neighbors:
        neighbor_dq = 0
        for n in neighbors:
            n_queue = all_new_metrics[n]["queue"]
            n_prev_q = prev_metrics[n]["queue"]
            neighbor_dq += (n_prev_q - n_queue) / (n_prev_q + 1)
        neighbor_avg_dq = (neighbor_dq / len(neighbors)) * 2.0
    else:
        neighbor_avg_dq = 0

    reward = (
        5.0 * delta_q +
        2.0 * delta_w +
        1.0 * neighbor_avg_dq -
        0.3 * abs(pressure) +
        throughput_reward
    )

    return reward, delta_q, delta_w, pressure, neighbor_avg_dq


# -------------------------------------------------------
# TRAINING STEP (paper Sec IV-F, steps 5-7)
# -------------------------------------------------------


def train_on_batch():
    if len(replay_buffer) < MIN_BUFFER_BEFORE_TRAIN:
        return None

    batch = replay_buffer.sample(BATCH_SIZE)

    state_batch = torch.as_tensor(
        [b[0] for b in batch], dtype=torch.float32, device=DEVICE
    )
    next_state_batch = torch.as_tensor(
        [b[4] for b in batch], dtype=torch.float32, device=DEVICE
    )
    reward_batch = torch.as_tensor(
        [b[3] for b in batch], dtype=torch.float32, device=DEVICE
    ).view(-1, 1)

    agent_q_columns = []
    target_q_columns = []

    for jid in JUNCTION_IDS:
        agent = agents[jid]

        obs_batch = torch.as_tensor(
            [b[1][jid] for b in batch], dtype=torch.float32, device=DEVICE
        )
        action_batch = torch.as_tensor(
            [b[2][jid] for b in batch], dtype=torch.long, device=DEVICE
        ).view(-1, 1)
        next_obs_batch = torch.as_tensor(
            [b[5][jid] for b in batch], dtype=torch.float32, device=DEVICE
        )

        q_values = agent.q_network(obs_batch)
        chosen_q = q_values.gather(1, action_batch)
        agent_q_columns.append(chosen_q)

        with torch.no_grad():
            next_q_values = agent.target_network(next_obs_batch)
            max_next_q = next_q_values.max(dim=1, keepdim=True)[0]
        target_q_columns.append(max_next_q)

    agent_qs = torch.cat(agent_q_columns, dim=1)                # (batch, num_agents)
    target_agent_qs = torch.cat(target_q_columns, dim=1)        # (batch, num_agents)

    q_tot = mixing_network(agent_qs, state_batch)
    with torch.no_grad():
        target_q_tot = target_mixing_network(target_agent_qs, next_state_batch)
        y = reward_batch + GAMMA * target_q_tot

    loss = F.mse_loss(q_tot, y)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(all_params, max_norm=5.0)
    optimizer.step()

    return loss.item()


def hard_update_targets():
    for agent in agents.values():
        agent.update_target()
    target_mixing_network.load_state_dict(mixing_network.state_dict())


# -------------------------------------------------------
# CSV INIT
# -------------------------------------------------------

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
        "teleports",
        "epsilon",
        "avg_td_loss",
    ])

with open(JUNCTION_LOG_FILE, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "episode",
        "junction",
        "avg_reward",
        "avg_queue",
        "avg_wait",
        "avg_pressure",
        "avg_delta_q",
        "avg_delta_w",
        ])


with open(ACTION_LOG_FILE, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "episode",
        "junction",
        "action",          # 0=hold, 1=extend, 2=switch
        "greedy_count",    # times this action was chosen by argmax(Q)
        "random_count",    # times this action was chosen by epsilon-explore
    ])

# -------------------------------------------------------
# TRAINING LOOP (paper Sec IV-F)
# -------------------------------------------------------

for episode in range(NUM_EPISODES):

    total_fuel_episode = 0.0
    total_queue_episode = 0
    total_wait_episode = 0
    step_count = 0
    step = 0
    total_pressure = 0
    total_delta_q = 0
    total_delta_w = 0
    total_teleports = 0
    episode_losses = []

    # per-junction accumulators, reset each episode (for the breakdown CSV)
    junction_reward_sum = {jid: 0.0 for jid in JUNCTION_IDS}
    junction_queue_sum = {jid: 0.0 for jid in JUNCTION_IDS}
    junction_wait_sum = {jid: 0.0 for jid in JUNCTION_IDS}
    junction_pressure_sum = {jid: 0.0 for jid in JUNCTION_IDS}
    junction_dq_sum = {jid: 0.0 for jid in JUNCTION_IDS}
    junction_dw_sum = {jid: 0.0 for jid in JUNCTION_IDS}

    # per-junction, per-action exploit/explore counts, reset each episode
    action_counts = {
        jid: {a: {"greedy": 0, "random": 0} for a in ACTIONS}
        for jid in JUNCTION_IDS
    }

    print(f"\n===== Episode {episode} (Stage 4 / QMIX) =====")

    for jid in JUNCTION_IDS:
        prev_metrics[jid] = {"queue": 0, "wait": 0}
        last_switch_time[jid] = 0
        phase_start_time[jid] = 0
        last_seen_phase[jid] = -1

    train_seed = random.choice(TRAIN_SEED_POOL)
    traci.start([SUMO_BINARY, "-c", CONFIG_FILE, "--seed", str(train_seed)])

    total_reward_episode = 0

    try:
        while traci.simulation.getTime() < MAX_SIM_TIME and step < MAX_STEPS:

            traci.simulationStep()
            now = traci.simulation.getTime()
            step_teleports = traci.simulation.getStartingTeleportNumber()
            if step_teleports > 0:
                print(f"[episode {episode}] [t={now:.0f}] {step_teleports} vehicle(s) started teleporting (via TraCI)")
            total_teleports += step_teleports
            step += 1

            # ---- observe (step 1) ----
            raw_before = {jid: get_raw_metrics(jid) for jid in JUNCTION_IDS}
            obs_before = build_all_obs(raw_before)
            state_before = build_global_state(raw_before)

            # ---- act (step 2-3) ----
            actions_taken = {}
            for jid in JUNCTION_IDS:
                action, was_greedy = agents[jid].choose_action(obs_before[jid])
                actions_taken[jid] = action
                apply_action(jid, action, now)

                key = "greedy" if was_greedy else "random"
                action_counts[jid][action][key] += 1

            # ---- SUMO already advanced this step; read resulting metrics ----
            new_metrics = {
                jid: {"queue": get_queue_length(jid), "wait": get_avg_waiting_time(jid)}
                for jid in JUNCTION_IDS
            }

            raw_after = {jid: get_raw_metrics(jid) for jid in JUNCTION_IDS}
            obs_after = build_all_obs(raw_after)
            state_after = build_global_state(raw_after)

            step_reward = 0
            for jid in JUNCTION_IDS:
                reward, dq, dw, pressure, ndq = compute_reward(
                    jid, new_metrics[jid], new_metrics
                )
                step_reward += reward
                total_delta_q += dq
                total_delta_w += dw
                total_pressure += pressure

                junction_reward_sum[jid] += reward
                junction_queue_sum[jid] += new_metrics[jid]["queue"]
                junction_wait_sum[jid] += new_metrics[jid]["wait"]
                junction_pressure_sum[jid] += pressure
                junction_dq_sum[jid] += dq
                junction_dw_sum[jid] += dw

            # ---- store joint transition (step 4) ----
            replay_buffer.push(
                state_before, obs_before, actions_taken,
                step_reward, state_after, obs_after
            )

            # ---- train (steps 5-6) ----
            loss = train_on_batch()
            if loss is not None:
                episode_losses.append(loss)

            train_step_counter += 1
            if train_step_counter % TARGET_UPDATE_EVERY_N_STEPS == 0:
                hard_update_targets()  # step 7

            for jid in JUNCTION_IDS:
                prev_metrics[jid] = new_metrics[jid]
                total_queue_episode += new_metrics[jid]["queue"]
                total_wait_episode += new_metrics[jid]["wait"]

            total_fuel_episode += get_total_fuel()
            step_count += 1
            total_reward_episode += step_reward

    finally:
        for agent in agents.values():
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay
        traci.close()

    avg_queue = total_queue_episode / step_count if step_count else 0
    avg_wait = total_wait_episode / step_count if step_count else 0
    avg_pressure = total_pressure / step_count if step_count else 0
    avg_dq = total_delta_q / step_count if step_count else 0
    avg_dw = total_delta_w / step_count if step_count else 0
    avg_loss = sum(episode_losses) / len(episode_losses) if episode_losses else 0

    epsilon_value = list(agents.values())[0].epsilon

    print(
        f"Teleports: {total_teleports} | Fuel: {total_fuel_episode:.2f} | "
        f"Epsilon: {epsilon_value:.3f} | Avg TD loss: {avg_loss:.4f}"
    )

    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            episode, total_reward_episode, total_fuel_episode,
            avg_queue, avg_wait, avg_pressure, avg_dq, avg_dw,
            total_teleports, epsilon_value, avg_loss,
        ])

    with open(JUNCTION_LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        for jid in JUNCTION_IDS:
            writer.writerow([
                episode,
                JUNCTION_LABELS[jid],
                junction_reward_sum[jid] / step_count if step_count else 0,
                junction_queue_sum[jid] / step_count if step_count else 0,
                junction_wait_sum[jid] / step_count if step_count else 0,
                junction_pressure_sum[jid] / step_count if step_count else 0,
                junction_dq_sum[jid] / step_count if step_count else 0,
                junction_dw_sum[jid] / step_count if step_count else 0,
            ])

    with open(ACTION_LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        for jid in JUNCTION_IDS:
            for a in ACTIONS:
                writer.writerow([
                    episode,
                    JUNCTION_LABELS[jid],
                    a,
                    action_counts[jid][a]["greedy"],
                    action_counts[jid][a]["random"],
                ])

print("\nStage 4 (QMIX) training complete!")

torch.save(
    {jid: agents[jid].q_network.state_dict() for jid in JUNCTION_IDS},
    os.path.join(BASE_DIR, "stage4_checkpoint.pt")
)
print("Saved trained Q-networks to stage4_checkpoint.pt")

# -------------------------------------------------------
# FINAL EVALUATION RUN (step 8: decentralized execution, epsilon=0,
# mixing network + global state discarded - Sec IV-E)
# -------------------------------------------------------

RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEBUG_ACTIONS = False  # keep console readable across multiple eval runs
DEBUG_MAX_EPISODES = 0

for agent in agents.values():
    agent.epsilon = 0.0  # pure exploitation via each agent's own Q-network

# Evaluate on several different SUMO seeds instead of one fixed seed.
# A single deterministic episode (fixed seed + epsilon=0) can land on a
# scenario that happens to expose one weak spot in an otherwise solid
# policy - averaging over several seeds gives a trustworthy estimate
# of how the policy performs in general, not how it did on one roll.
EVAL_SEEDS = [42, 123, 7, 99, 256]
eval_results = []

from utils.metrics_utils import summarize_tripinfo, summarize_summary_output

JN_TRACE_FILE = os.path.join(RESULTS_DIR, "jn_trace_seed42.csv")
with open(JN_TRACE_FILE, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time", "phase", "queue", "wait", "pressure", "action"])
    
for seed in EVAL_SEEDS:
    TRIPINFO_FILE = os.path.join(RESULTS_DIR, f"tripinfo_stage4_seed{seed}.xml")
    SUMMARY_FILE = os.path.join(RESULTS_DIR, f"summary_stage4_seed{seed}.xml")

    for jid in JUNCTION_IDS:
        prev_metrics[jid] = {"queue": 0, "wait": 0}
        last_switch_time[jid] = 0
        phase_start_time[jid] = 0
        last_seen_phase[jid] = -1

    traci.start([
        SUMO_BINARY, "-c", CONFIG_FILE,
        "--seed", str(seed),
        "--tripinfo-output", TRIPINFO_FILE,
        "--summary-output", SUMMARY_FILE,
    ])

    eval_fuel = 0.0
    step = 0
    try:
        while traci.simulation.getTime() < MAX_SIM_TIME and step < MAX_STEPS:
            traci.simulationStep()
            now = traci.simulation.getTime()
            step += 1

            raw = {jid: get_raw_metrics(jid) for jid in JUNCTION_IDS}
            obs = build_all_obs(raw)

            for jid in JUNCTION_IDS:
                action, _ = agents[jid].choose_action(obs[jid])
                apply_action(jid, action, now)

                if jid == JN_ID and seed == 42:
                    with open(JN_TRACE_FILE, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            round(now, 1),
                            raw[jid]["phase"],
                            raw[jid]["queue"],
                            round(raw[jid]["wait"], 2),
                            round(get_pressure(jid), 4),
                            action,
                        ])

            eval_fuel += get_total_fuel()
    finally:
        traci.close()

    avg_wait_e, avg_travel_e, throughput_e, avg_stops_e = summarize_tripinfo(TRIPINFO_FILE)
    avg_queue_e, avg_speed_e = summarize_summary_output(SUMMARY_FILE)

    print(
        f"[eval seed={seed}] wait={avg_wait_e:.2f} queue={avg_queue_e:.2f} "
        f"travel={avg_travel_e:.2f} throughput={throughput_e} "
        f"speed={avg_speed_e:.2f} stops={avg_stops_e:.2f} fuel={eval_fuel:.2f}"
    )

    eval_results.append({
        "seed": seed,
        "avg_waiting_time": avg_wait_e,
        "avg_queue_length": avg_queue_e,
        "avg_travel_time": avg_travel_e,
        "throughput": throughput_e,
        "avg_speed": avg_speed_e,
        "num_stops": avg_stops_e,
        "fuel_consumption": eval_fuel,
    })

with open("stage4_summary_per_seed.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "seed", "avg_waiting_time", "avg_queue_length", "avg_travel_time",
        "throughput", "avg_speed", "num_stops", "fuel_consumption"
    ])
    for r in eval_results:
        writer.writerow([
            r["seed"], round(r["avg_waiting_time"], 2), round(r["avg_queue_length"], 2),
            round(r["avg_travel_time"], 2), r["throughput"], round(r["avg_speed"], 2),
            round(r["num_stops"], 2), round(r["fuel_consumption"], 2),
        ])

n = len(eval_results)
avg_wait_final = sum(r["avg_waiting_time"] for r in eval_results) / n
avg_queue_final = sum(r["avg_queue_length"] for r in eval_results) / n
avg_travel_final = sum(r["avg_travel_time"] for r in eval_results) / n
avg_throughput_final = sum(r["throughput"] for r in eval_results) / n
avg_speed_final = sum(r["avg_speed"] for r in eval_results) / n
avg_stops_final = sum(r["num_stops"] for r in eval_results) / n
avg_fuel_final = sum(r["fuel_consumption"] for r in eval_results) / n

with open("stage4_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "avg_waiting_time", "avg_queue_length", "avg_travel_time",
        "throughput", "avg_speed", "num_stops", "fuel_consumption"
    ])
    writer.writerow([
        round(avg_wait_final, 2), round(avg_queue_final, 2), round(avg_travel_final, 2),
        round(avg_throughput_final, 1), round(avg_speed_final, 2), round(avg_stops_final, 2),
        round(avg_fuel_final, 2)
    ])

print(f"\nStage 4 evaluation summary (averaged over {n} seeds) written to stage4_summary.csv")
print(f"Per-seed breakdown written to stage4_summary_per_seed.csv")