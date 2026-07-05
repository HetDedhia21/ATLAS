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

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please set the SUMO_HOME environment variable.")

import traci

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

SUMO_BINARY = "sumo-gui"
CONFIG_FILE = "smart5_adaptive.sumocfg"

JUNCTION_IDS = [
    "cluster13437517362_4374680526_4374680531_5346620503_#2more",
    "cluster1936499662_2629436073_621284871",
    "cluster6616421101_6616421102",
    "cluster1646746616_6430722657_7263291410",
    "cluster1156127277_5346644809",
]

# -------------------------------------------------------
# CONTROL PARAMETERS
# -------------------------------------------------------

QUEUE_THRESHOLD = 5
EXTENSION_SECONDS = 5.0
MIN_PHASE_DURATION = 10.0

MAX_PHASE_DURATION = 60.0
ABSOLUTE_MAX_PHASE_DURATION = 120.0

DECISION_INTERVAL = 5.0

phase_start_time = {jid: 0.0 for jid in JUNCTION_IDS}
last_decision_time = {jid: 0.0 for jid in JUNCTION_IDS}


# -------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------

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
        for veh in traci.lane.getLastStepVehicleIDs(lane):
            waits.append(traci.vehicle.getWaitingTime(veh))

    return sum(waits) / len(waits) if waits else 0.0


def get_lanes_served_by_current_phase(junction_id):

    controlled_lanes = traci.trafficlight.getControlledLanes(junction_id)

    state = traci.trafficlight.getRedYellowGreenState(junction_id)

    green_lanes = set()

    for lane, signal in zip(controlled_lanes, state):
        if signal in ("g", "G"):
            green_lanes.add(lane)

    return green_lanes


def get_congested_lanes(junction_id, threshold=2):

    congested = set()

    lanes = set(traci.trafficlight.getControlledLanes(junction_id))

    for lane in lanes:
        if traci.lane.getLastStepHaltingNumber(lane) > threshold:
            congested.add(lane)

    return congested


# -------------------------------------------------------
# ADAPTIVE CONTROLLER
# -------------------------------------------------------

def apply_rule_based_control(junction_id, now):

    if now - last_decision_time[junction_id] < DECISION_INTERVAL:
        return

    last_decision_time[junction_id] = now

    time_in_phase = now - phase_start_time[junction_id]

    # =====================================================
    # ABSOLUTE SAFETY LIMIT
    # =====================================================

    if time_in_phase >= ABSOLUTE_MAX_PHASE_DURATION:

        num_phases = len(
            traci.trafficlight.getAllProgramLogics(junction_id)[0].phases
        )

        current_phase = traci.trafficlight.getPhase(junction_id)

        traci.trafficlight.setPhase(
            junction_id,
            (current_phase + 1) % num_phases
        )

        phase_start_time[junction_id] = now

        return

    # =====================================================
    # SOFT LIMIT
    # =====================================================

    if time_in_phase >= MAX_PHASE_DURATION:

        congested_lanes_check = get_congested_lanes(
            junction_id,
            threshold=2
        )

        green_lanes_check = get_lanes_served_by_current_phase(
            junction_id
        )

        if not (congested_lanes_check & green_lanes_check):

            num_phases = len(
                traci.trafficlight.getAllProgramLogics(junction_id)[0].phases
            )

            current_phase = traci.trafficlight.getPhase(junction_id)

            traci.trafficlight.setPhase(
                junction_id,
                (current_phase + 1) % num_phases
            )

            phase_start_time[junction_id] = now

        return

    congested_lanes = get_congested_lanes(
        junction_id,
        threshold=2
    )

    green_lanes = get_lanes_served_by_current_phase(
        junction_id
    )

    # ----------------------------------------------------
    # DEBUG OUTPUT
    # ----------------------------------------------------

    if junction_id == "cluster13437517362_4374680526_4374680531_5346620503_#2more":

        occ = [
            traci.lane.getLastStepVehicleNumber(l)
            for l in green_lanes
        ]

        print(f"  [debug] t={now:.1f}")
        print(f"  [debug] green_lanes={green_lanes}")
        print(f"  [debug] occupancy={occ}")

    green_lane_occupancy = [
        traci.lane.getLastStepVehicleNumber(l)
        for l in green_lanes
    ]

    phase_serves_no_one = (
        len(green_lanes) > 0
        and
        sum(green_lane_occupancy) == 0
    )

    if junction_id == "cluster13437517362_4374680526_4374680531_5346620503_#2more":

        print(
            f"  [debug] time={time_in_phase:.1f}"
        )

        print(
            f"  [debug] empty={phase_serves_no_one}"
        )

        print(
            f"  [debug] congested={congested_lanes}"
        )

        print(
            f"  [debug] helping={bool(congested_lanes & green_lanes)}"
        )

    if not congested_lanes and not phase_serves_no_one:
        return

    currently_helping = bool(
        congested_lanes & green_lanes
    )

    if (
        phase_serves_no_one
        and
        time_in_phase >= MIN_PHASE_DURATION
    ):

        num_phases = len(
            traci.trafficlight.getAllProgramLogics(junction_id)[0].phases
        )

        current_phase = traci.trafficlight.getPhase(junction_id)

        traci.trafficlight.setPhase(
            junction_id,
            (current_phase + 1) % num_phases
        )

        phase_start_time[junction_id] = now

        return

    if not congested_lanes:
        return

    if currently_helping:

        if time_in_phase >= MIN_PHASE_DURATION:

            remaining = (
                traci.trafficlight.getNextSwitch(junction_id)
                - now
            )

            traci.trafficlight.setPhaseDuration(
                junction_id,
                remaining + EXTENSION_SECONDS
            )

    else:

        if time_in_phase >= MIN_PHASE_DURATION:

            num_phases = len(
                traci.trafficlight.getAllProgramLogics(junction_id)[0].phases
            )

            current_phase = traci.trafficlight.getPhase(junction_id)

            traci.trafficlight.setPhase(
                junction_id,
                (current_phase + 1) % num_phases
            )

            phase_start_time[junction_id] = now

        return
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

    # initialize timers
    for jid in JUNCTION_IDS:
        phase_start_time[jid] = traci.simulation.getTime()
        last_decision_time[jid] = traci.simulation.getTime()

    step = 0

    try:

        while traci.simulation.getMinExpectedNumber() > 0:

            traci.simulationStep()

            now = traci.simulation.getTime()

            # -----------------------------------------
            # Run adaptive controller
            # -----------------------------------------

            for jid in JUNCTION_IDS:
                apply_rule_based_control(jid, now)

            # -----------------------------------------
            # Console statistics
            # -----------------------------------------

            if step % 50 == 0:

                print(f"\n========== t = {now:.1f} s ==========")

                total_queue = 0
                total_wait = 0

                for jid in JUNCTION_IDS:

                    queue = get_queue_length(jid)
                    wait = get_avg_waiting_time(jid)

                    total_queue += queue
                    total_wait += wait

                    print(
                        f"{jid}\n"
                        f"   Queue      : {queue}\n"
                        f"   Avg Wait   : {wait:.1f} sec"
                    )

                print("--------------------------------------")
                print(f"Network Queue : {total_queue}")
                print(f"Average Wait  : {total_wait/len(JUNCTION_IDS):.1f} sec")
                print("--------------------------------------")

            step += 1

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")

    finally:
        traci.close()


if __name__ == "__main__":
    main()
