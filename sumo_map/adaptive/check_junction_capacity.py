"""
check_junction_capacity.py

Verifies (rather than guesses) whether a junction's incoming demand
exceeds what its incoming lanes can plausibly handle, by:

  1. Reading the real SUMO net file to get each junction's actual
     incoming edges and lane counts (via sumolib, not ID-string matching).
  2. Reading the route/flow file's vehsPerHour flows.
  3. For every flow, computing its shortest path through the network and
     checking which of our 5 junctions that path passes through.
  4. Summing vehsPerHour-through-junction per junction, and dividing by
     that junction's incoming lane count, to get a rough veh/hr/lane
     load figure comparable across junctions.

Run from anywhere; paths default to the layout used by
sumo_map/adaptive/smart5_adaptive.sumocfg. Override with --net/--routes
if your paths differ.

Requires SUMO_HOME to be set (so sumolib is importable) - same
requirement as traci itself.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

# --- make sumolib importable, same convention traci scripts already use ---
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit(
        "SUMO_HOME is not set. Set it the same way you do before running "
        "traci_control_stage4.py, e.g.:\n"
        "  export SUMO_HOME=/usr/share/sumo   (Linux, apt install)\n"
        "  export SUMO_HOME=/opt/homebrew/opt/sumo/share/sumo   (macOS brew)\n"
    )

import sumolib  # noqa: E402

# --- junction IDs, same as traci_control_stage4.py JUNCTION_IDS ---
JUNCTIONS = {
    "J0_central": "cluster13437517362_4374680526_4374680531_5346620503_#2more",
    "JN_north": "cluster1936499662_2629436073_621284871",
    "JS_south": "cluster6616421101_6616421102",
    "JW_west": "cluster1646746616_6430722657_7263291410",
    "JE_east": "cluster1156127277_5346644809",
}


def parse_args():
    # script is meant to sit in sumo_map/adaptive/, right next to
    # smart5_adaptive.sumocfg, which itself points at "../network/..." -
    # so we resolve the same way the sumocfg does.
    here = os.path.dirname(os.path.abspath(__file__))
    default_net = os.path.join(here, "..", "network", "smart5_main.net.xml")
    default_routes = os.path.join(here, "..", "network", "smart5_flows.rou.xml")
    p = argparse.ArgumentParser()
    p.add_argument("--net", default=default_net,
                    help="path to smart5_main.net.xml (default: %(default)s)")
    p.add_argument("--routes", default=default_routes,
                    help="path to smart5_flows.rou.xml (default: %(default)s)")
    return p.parse_args()


def load_flows(routes_path):
    """Return list of (id, from_edge_id, to_edge_id, vehsPerHour) for every <flow>."""
    tree = ET.parse(routes_path)
    flows = []
    for flow in tree.getroot().findall("flow"):
        vph = flow.get("vehsPerHour")
        if vph is None:
            continue  # skip flows defined by 'number' or 'period' instead, if any
        flows.append((flow.get("id"), flow.get("from"), flow.get("to"), float(vph)))
    return flows


def main():
    args = parse_args()

    if not os.path.exists(args.net):
        sys.exit(f"Net file not found: {args.net}")
    if not os.path.exists(args.routes):
        sys.exit(f"Routes file not found: {args.routes}")

    print(f"Loading net: {args.net}")
    net = sumolib.net.readNet(args.net)

    # incoming edges + lane counts per junction, from real connectivity
    junction_incoming = {}
    junction_lanes = {}
    for label, node_id in JUNCTIONS.items():
        node = net.getNode(node_id)
        if node is None:
            print(f"  WARNING: junction id not found in net: {label} ({node_id})")
            continue
        incoming_edges = [e for e in node.getIncoming() if not e.isSpecial()]
        junction_incoming[label] = incoming_edges
        junction_lanes[label] = sum(e.getLaneNumber() for e in incoming_edges)

    print()
    print("--- Incoming lanes per junction (from actual net connectivity) ---")
    for label in JUNCTIONS:
        if label not in junction_incoming:
            continue
        edge_ids = [e.getID() for e in junction_incoming[label]]
        print(f"{label:12s} lanes={junction_lanes[label]:2d}  edges={edge_ids}")

    print()
    print("Loading flows and tracing shortest path for each...")
    flows = load_flows(args.routes)

    demand_through = defaultdict(float)   # label -> summed vehsPerHour passing through
    demand_per_edge = defaultdict(float)  # individual incoming edge id -> summed vehsPerHour
    unrouted = []

    for fid, from_id, to_id, vph in flows:
        from_edge = net.getEdge(from_id) if from_id else None
        to_edge = net.getEdge(to_id) if to_id else None
        if from_edge is None or to_edge is None:
            unrouted.append((fid, from_id, to_id, "edge id not found in net"))
            continue

        path_edges, _cost = net.getShortestPath(from_edge, to_edge)
        if path_edges is None:
            unrouted.append((fid, from_id, to_id, "no path found"))
            continue

        path_edge_ids = {e.getID() for e in path_edges}

        for label, incoming_edges in junction_incoming.items():
            incoming_ids = {e.getID() for e in incoming_edges}
            # flow passes "through" this junction if its path uses one of the
            # junction's incoming edges (i.e. it arrives at this junction)
            if path_edge_ids & incoming_ids:
                demand_through[label] += vph
            # also credit each specific incoming edge the path actually uses,
            # so we can see how demand splits *between* a junction's own
            # approaches, not just the junction-level total
            for eid in incoming_ids:
                if eid in path_edge_ids:
                    demand_per_edge[eid] += vph

    print()
    print("--- Demand vs. capacity per junction ---")
    print(f"{'Junction':12s} {'lanes':>6s} {'veh/hr thru':>12s} {'veh/hr/lane':>12s}")
    for label in JUNCTIONS:
        if label not in junction_lanes or junction_lanes[label] == 0:
            continue
        lanes = junction_lanes[label]
        vph = demand_through.get(label, 0.0)
        per_lane = vph / lanes
        print(f"{label:12s} {lanes:6d} {vph:12.0f} {per_lane:12.1f}")

    print()
    print("--- Demand split between each junction's OWN incoming edges ---")
    print("(this is the number that matters for whether a fixed 50/50 phase")
    print(" split under- or over-serves one of the two approaches)")
    for label in JUNCTIONS:
        if label not in junction_incoming:
            continue
        edges = junction_incoming[label]
        if len(edges) < 2:
            continue
        rows = sorted(((e.getID(), demand_per_edge.get(e.getID(), 0.0)) for e in edges),
                       key=lambda r: -r[1])
        total = sum(v for _, v in rows) or 1.0
        print(f"{label}:")
        for eid, v in rows:
            print(f"    {eid:20s} {v:8.0f} veh/hr  ({100*v/total:5.1f}%)")

    if unrouted:
        print()
        print(f"--- {len(unrouted)} flow(s) could not be routed/matched (informational) ---")
        for fid, f, t, reason in unrouted:
            print(f"  {fid}: from={f} to={t}  ({reason})")

    print()
    print("Read the veh/hr/lane column: if JN_north's figure is well above")
    print("JS_south's despite similar lane counts, that supports the demand/")
    print("capacity mismatch hypothesis rather than an RL/reward bug.")


if __name__ == "__main__":
    main()