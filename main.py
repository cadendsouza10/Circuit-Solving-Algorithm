import numpy as np

# --------------------------------------------------
# Read Circuit File
# --------------------------------------------------

components = []

with open("circuit.txt", "r") as file:
    for line in file:

        line = line.strip()
        
        # ignore blanks
        if not line:
            continue

        parts = line.split()

        name = parts[0]
        node1 = int(parts[1])
        node2 = int(parts[2])
        value = float(parts[3])

        components.append({
            "name" : name,
            "node1": node1,
            "node2": node2,
            "value" : value
        })


# --------------------------------------------------
# Find Amount of Nodes
# --------------------------------------------------


max_node = 0

for component in components:
    max_node = max(max_node, component["node1"], component["node2"])

num_nodes = max_node



# --------------------------------------------------
# Find Voltage Sources
# --------------------------------------------------

voltage_sources = []

for component in components:
    if component["name"].startswith("V"):
        voltage_sources.append(component)

num_voltage_sources = len(voltage_sources)



# --------------------------------------------------
# Create MNA Matrix
# --------------------------------------------------

# Unknowns:
#
# [V1, V2, V3, ..., I_V1, I_V2, ...]
#
# Node 0 is ground, so it is NOT included
# as an unknown.


matrix_size = num_nodes + num_voltage_sources

A = np.zeros((matrix_size, matrix_size))
z = np.zeros(matrix_size)


# --------------------------------------------------
# Stamp Resistors
# --------------------------------------------------

for component in components:
    if component["name"].startswith("R"):
        node1 = component["node1"]
        node2 = component["node2"]
        R = component["value"]
        conductance = 1 / R

        if node1 != 0:
            A[node1 - 1, node1 - 1] += conductance

        if node2 != 0:
            A[node2 - 1, node2 - 1] += conductance

        if node1 != 0 and node2 != 0:
            A[node1 - 1, node2 - 1] -= conductance
            A[node2 - 1, node1 - 1] -= conductance



# --------------------------------------------------
# Stamp Voltage Sources
# --------------------------------------------------

for index, source in enumerate(voltage_sources):
    node1 = source["node1"]
    node2 = source["node2"]
    voltage = source["value"]

    # Current unknown index for this voltage source
    source_index = num_nodes + index

    if node1 != 0:
        A[node1 - 1, source_index] += 1
        A[source_index, node1 - 1] += 1

    if node2 != 0:
        A[node2 - 1, source_index] -= 1
        A[source_index, node2 - 1] -= 1

    z[source_index] = voltage


# --------------------------------------------------
# Solve The Circuit
# --------------------------------------------------


solution = np.linalg.solve(A, z)



# --------------------------------------------------
# Print Node Voltages
# --------------------------------------------------


print("\n===================================")
print("       PYTHON CIRCUIT SIMULATOR")
print("===================================")

print("\nNODE VOLTAGES")
print("-----------------------------------")

print("Node 0 (GND): 0.000 V")

for node in range(1, num_nodes + 1):
    voltage = solution[node - 1]
    print(f"Node {node}: {voltage:.3f} V")



# --------------------------------------------------
# Print Component Currents
# --------------------------------------------------

for component in components:

    if component["name"].startswith("R"):

        node1 = component["node1"]
        node2 = component["node2"]
        R = component["value"]

        if node1 == 0:
            V1 = 0
        else:
            V1 = solution[node1 - 1]

        if node2 == 0:
            V2 = 0
        else:
            V2 = solution[node2 - 1]

        current = (V1 - V2) / R

        print(
            f"{component['name']}: "
            f"{current * 1000:.3f} mA"
        )


# --------------------------------------------------
# Print Resistor Power
# --------------------------------------------------


print("\nRESISTOR POWER")
print("-----------------------------------")


for component in components:

    if component["name"].startswith("R"):

        node1 = component["node1"]
        node2 = component["node2"]
        R = component["value"]

        if node1 == 0:
            V1 = 0
        else:
            V1 = solution[node1 - 1]

        if node2 == 0:
            V2 = 0
        else:
            V2 = solution[node2 - 1]

        voltage_drop = V1 - V2

        power = (voltage_drop ** 2) / R

        print(
            f"{component['name']}: "
            f"{power * 1000:.3f} mW"
        )


# --------------------------------------------------
# Print Voltage Source Currents
# --------------------------------------------------

print("\nVOLTAGE SOURCE CURRENTS")
print("-----------------------------------")

for index, source in enumerate(voltage_sources):

    source_current = solution[num_nodes + index]

    print(
        f"{source['name']}: "
        f"{source_current * 1000:.3f} mA"
    )