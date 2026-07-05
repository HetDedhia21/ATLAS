import xml.etree.ElementTree as ET
import random

tree = ET.parse("routes.rou.xml")
root = tree.getroot()

vehicle_types = [
    ("car", 50),
    ("motorbike", 25),
    ("rickshaw", 10),
    ("bus", 7),
    ("truck", 8)
]

population = []
for vtype, count in vehicle_types:
    population.extend([vtype] * count)

for vehicle in root.findall("vehicle"):
    vehicle.set("type", random.choice(population))

tree.write(
    "routes.rou.xml",
    encoding="UTF-8",
    xml_declaration=True
)

print("Done! Vehicle types assigned successfully.")
