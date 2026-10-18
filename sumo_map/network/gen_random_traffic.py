"""Generate uniform random traffic for every vehicle type with SUMO's randomTrips.py.

Save this file inside your network/ folder and run:  py gen_random_traffic.py
Outputs (rand_*.trips.xml and rand_*.rou.xml) are written next to this script.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)  # outputs always land in the same folder as this script

NET = "smart5_main_updated.net.xml"
VTYPES = "vtypes.add.xml"  # optional; used if present so --validate knows the types
END = 3600

sumo_home = os.environ.get("SUMO_HOME")
if not sumo_home:
    sys.exit("SUMO_HOME is not set.")
tool = os.path.join(sumo_home, "tools", "randomTrips.py")

if not os.path.exists(NET):
    sys.exit(f"Net file not found in {HERE}: {NET}")
    
LEVEL = float(sys.argv[1]) if len(sys.argv) > 1 else 5
if not 1 <= LEVEL <= 10:
    sys.exit("Level must be between 1 and 10.")
SCALE = LEVEL / 10.0

# name: (vType id, seconds between vehicles for the whole network)
TYPES = {
    "car":   ("car",       1.5),
    "bike":  ("motorbike", 1.0),
    "rick":  ("rickshaw",  4.0),
    "bus":   ("bus",       12.0),
    "truck": ("truck",     10.0),
}

for i, (name, (vtype, period)) in enumerate(TYPES.items()):
    cmd = [
        sys.executable, tool,
        "-n", NET,
        "-b", "0", "-e", str(END), "-p", str(round(period / SCALE, 3)),
        "--prefix", f"{name}_",
        "--seed", str(42 + i),
        "--trip-attributes", f'type="{vtype}" departLane="best" departSpeed="max"',
        "--random-departpos",
        "--min-distance", "300",
        "--validate",
        "-o", f"rand_{name}.trips.xml",
        "-r", f"rand_{name}.rou.xml",
    ]
    if os.path.exists(VTYPES):
        cmd += ["--additional-files", VTYPES]
    print(f"\n=== {name} ===")
    subprocess.run(cmd, check=True)

print("\nDone. Files created in:", HERE)
for f in sorted(os.listdir(HERE)):
    if f.startswith("rand_"):
        print("  ", f)