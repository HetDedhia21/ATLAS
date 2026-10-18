"""Generate random ambulance trips across the whole network.

Save in network/ next to the net file and run:  py gen_emergency.py
Output: rand_ambulance.rou.xml  (add this to <route-files> in your .sumocfg)

The ambulance vType is written into emergency_vtype.add.xml (used only for
validation here). duarouter copies the vType into the output route file, so do
NOT also list emergency_vtype.add.xml in the config (that causes the
"Another vehicle type with the id ... exists" error).
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

NET = "smart5_main_updated.net.xml"
VTYPE_FILE = "emergency_vtype.add.xml"
END = 3600
PERIOD = 120.0  # one ambulance every 120 s on average (~30/hr); raise for fewer

VTYPE_XML = """<additional>
    <vType id="ambulance" vClass="emergency" guiShape="emergency" color="white"
           length="6.00" minGap="1.50" maxSpeed="25.00" speedFactor="1.3"
           accel="3.0" decel="5.0" sigma="0.2">
        <param key="has.bluelight.device" value="true"/>
        <param key="device.bluelight.reactiondist" value="50"/>
    </vType>
</additional>
"""

with open(VTYPE_FILE, "w") as f:
    f.write(VTYPE_XML)

sumo_home = os.environ.get("SUMO_HOME")
if not sumo_home:
    sys.exit("SUMO_HOME is not set.")
tool = os.path.join(sumo_home, "tools", "randomTrips.py")
if not os.path.exists(NET):
    sys.exit(f"Net file not found in {HERE}: {NET}")

cmd = [
    sys.executable, tool,
    "-n", NET,
    "-b", "0", "-e", str(END), "-p", str(PERIOD),
    "--prefix", "amb_",
    "--seed", "99",
    "--trip-attributes", 'type="ambulance" departLane="best" departSpeed="max"',
    "--min-distance", "1000",   # long trips so they cross several junctions
    "--fringe-factor", "10",    # start/end near the network boundary
    "--validate",
    "--additional-files", VTYPE_FILE,
    "-o", "rand_ambulance.trips.xml",
    "-r", "rand_ambulance.rou.xml",
]
subprocess.run(cmd, check=True)
print("\nDone:", os.path.join(HERE, "rand_ambulance.rou.xml"))