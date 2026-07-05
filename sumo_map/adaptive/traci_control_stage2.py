"""
ATLAS - Stage 2: Rule-based adaptive signal control
Same state-reading as Stage 1, but now actively extends green phases at
junctions with heavy queues instead of just reporting them. This is a
placeholder "smart" controller so we can validate the full before/after
comparison pipeline before the MARL agents are trained (Stage 3 will swap
the decision logic below for trained policies, without touching the rest
of this script).
"""

import os
import sys
import traci
import csv

LOG_FILE = "stage2_metrics.csv"
MAX_SIM_TIME = 400


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please set the SUMO_HOME environment variable (see earlier setup).")

# ---- PATH SETUP (FIXED) ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SUMO_BINARY = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo-gui.exe")
CONFIG_FILE = os.path.join(BASE_DIR, "smart5_adaptive.sumocfg")

JUNCTION_IDS = [
    "cluster13437517362_4374680526_4374680531_5346620503_#2more",  # main 5-way
    "cluster1936499662_2629436073_621284871",                       # north
    "cluster6616421101_6616421102",                                 # south
    "cluster1646746616_6430722657_7263291410",                      # west
    "cluster1156127277_5346644809",                                 # east
]

# ---- CONTROL PARAMETERS ----
QUEUE_THRESHOLD = 5        # if halted vehicles on the active approach exceed this, extend green
EXTENSION_SECONDS = 5.0    # how much to extend the current green phase by, per decision
MIN_PHASE_DURATION = 10.0  # never let a phase run shorter than this (avoids flicker)
MAX_PHASE_DURATION = 60.0  # hard cap so one direction can't hog green forever
DECISION_INTERVAL = 5.0    # how often (sim seconds) each junction re-evaluates its decision

# tracks, per junction, when its current phase started (sim time) so we know
# how long it's been running and whether we're allowed to extend/switch yet
phase_start_time = {jid: 0.0 for jid in JUNCTION_IDS}
last_decision_time = {jid: 0.0 for jid in JUNCTION_IDS}


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
        for v in traci.lane.getLastStepVehicleIDs(lane):
            waits.append(traci.vehicle.getWaitingTime(v))
    return sum(waits) / len(waits) if waits else 0.0


def get_lanes_served_by_current_phase(junction_id):
    """
    Returns the set of controlled lanes that actually have a green ('g' or 'G')
    signal in the junction's CURRENT phase state string. This is the piece
    Stage 2 was missing - without it we can't tell whether the active phase
    is the one that would actually help a congested approach.
    """
    controlled_lanes = traci.trafficlight.getControlledLanes(junction_id)
    state = traci.trafficlight.getRedYellowGreenState(junction_id)
    # state is a string like "GGrrGGrr" - one character per controlled link,
    # same order as getControlledLanes (though a lane can appear more than once
    # if it has multiple links, hence zip rather than assuming a 1:1 set)
    green_lanes = set()
    for lane, signal_char in zip(controlled_lanes, state):
        if signal_char in ("g", "G"):
            green_lanes.add(lane)
    return green_lanes


def get_congested_lanes(junction_id, threshold=2):
    """Lanes at this junction with more than `threshold` halted vehicles."""
    lanes = set(traci.trafficlight.getControlledLanes(junction_id))
    congested = set()
    for lane in lanes:
        if traci.lane.getLastStepHaltingNumber(lane) > threshold:
            congested.add(lane)
    return congested


def apply_rule_based_control(junction_id, now):
    """
    Threshold rule, now direction-aware:
      - only extend the current green phase if it is actually serving at
        least one congested lane (otherwise extending it just wastes time
        on a direction that doesn't need it, which was the Stage 2 bug)
      - if the current phase does NOT serve the congested lanes, and it has
        already met MIN_PHASE_DURATION, switch to the next phase early so
        the congested direction gets served sooner instead of waiting out
        a fixed cycle
      - if a phase has run too long (MAX_PHASE_DURATION), force a switch
        regardless, so no direction can starve the others
    """
    if now - last_decision_time[junction_id] < DECISION_INTERVAL:
        return
    last_decision_time[junction_id] = now

    time_in_phase = now - phase_start_time[junction_id]

    if time_in_phase >= MAX_PHASE_DURATION:
        num_phases = len(traci.trafficlight.getAllProgramLogics(junction_id)[0].phases)
        current_phase = traci.trafficlight.getPhase(junction_id)
        traci.trafficlight.setPhase(junction_id, (current_phase + 1) % num_phases)
        phase_start_time[junction_id] = now
        return

    congested_lanes = get_congested_lanes(junction_id, threshold=2)
    if not congested_lanes:
        return  # nothing waiting badly enough to act on

    green_lanes = get_lanes_served_by_current_phase(junction_id)
    currently_helping = bool(congested_lanes & green_lanes)

    if currently_helping:
        # the active green is serving a congested lane - worth extending
        if time_in_phase >= MIN_PHASE_DURATION:
            remaining = traci.trafficlight.getNextSwitch(junction_id) - now
            traci.trafficlight.setPhaseDuration(junction_id, remaining + EXTENSION_SECONDS)
    else:
        # the active green is NOT helping the congested lanes - move on early
        # instead of extending irrelevant green time (this is the fix)
        if time_in_phase >= MIN_PHASE_DURATION:
            num_phases = len(traci.trafficlight.getAllProgramLogics(junction_id)[0].phases)
            current_phase = traci.trafficlight.getPhase(junction_id)
            traci.trafficlight.setPhase(junction_id, (current_phase + 1) % num_phases)
            phase_start_time[junction_id] = now


def main():
    sumo_cmd = [SUMO_BINARY, "-c", CONFIG_FILE]
    traci.start(sumo_cmd)

    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time",
            "junction_id",
            "queue_length",
            "avg_waiting_time"
    ])

    # initialize phase_start_time properly now that the sim has actually started
    for jid in JUNCTION_IDS:
        phase_start_time[jid] = traci.simulation.getTime()

    step = 0
    try:
        while (
            traci.simulation.getMinExpectedNumber() > 0
            and traci.simulation.getTime() < MAX_SIM_TIME
        ):
            traci.simulationStep()

            now = traci.simulation.getTime()

            for jid in JUNCTION_IDS:
                apply_rule_based_control(jid, now)

            # ✅ CSV LOGGING (NEW)
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
