import csv
import os
import sys

import numpy as np
import math

import mip


def main():
    # Check command line arguments
    if len(sys.argv) != 2:
        print(f"Invalid number of arguments.")
        sys.exit(1)

    csv_path = sys.argv[1]

    # Decide which building we have from the filename
    name = os.path.basename(csv_path).lower()
    if "1" in name:
        base = (0.0, -16.0, 0.0)
        entry_y_threshold = -12.5
    else:
        base = (0.0, -40.0, 0.0)
        entry_y_threshold = -20.0

    coords = {0: base}  # dictionary: node_id -> (x, y, z)

    with open(csv_path, newline="") as f:
        filtered = (line for line in f if not line.lstrip().startswith("#"))  # skip comment lines
        reader = csv.reader(filtered)
        _ = next(reader, None)  # remove the line with ["x", "y", "z"] labels

        for idx, row in enumerate(reader, start=1):
            if not row:  # skip empty lines
                continue
            x, y, z = map(float, row[:3])
            coords[idx] = (x, y, z)

    # At this point I have coords (vertex id -> (x,y,z)), I need to compute the edges and their costs
    # Define connection rules
    def axis_past_threshold(a, b):
        res = 0
        for i in range(len(a)):
            if abs(a[i] - b[i]) > 0.5:
                res += 1
        return res

    def connected(vertex_a, vertex_b):
        a = np.array(coords[vertex_a])
        b = np.array(coords[vertex_b])

        # Checks on entry points
        if vertex_a == 0 and vertex_b != 0 and b[1] <= entry_y_threshold:  # i == 0 corresponds to a being the base
            return True
        elif vertex_b == 0 and vertex_a != 0 and a[1] <= entry_y_threshold:  # j == 0 corresponds to b being the base
            return True
        elif vertex_a == vertex_b:
            return False

        d = np.linalg.norm(a - b)

        # Checks on normal grid points
        if d <= 4:
            return True
        elif (d <= 11) and (axis_past_threshold(a, b) <= 1):
            return True
        return False

    def compute_cost(i, j):
        a = coords[i]
        b = coords[j]

        horizontal_dist = math.sqrt((a[0] - b[0]) ** 2 + (a[2] - b[2]) ** 2)
        vertical_dist = abs(a[1] - b[1])

        return max(horizontal_dist / 1.5, vertical_dist / (1 if a[1] < b[1] else 2))

    V = list(coords.keys())  # nodes: 0...N, 0 is base
    arcs = []
    cost = {}

    # We have arcs and not edges because there is a difference in cost going up or down
    for i in V:
        for j in V:
            if i == j:
                continue
            if connected(i, j):
                arcs.append((i, j))
                cost[(i, j)] = compute_cost(i, j)

    # Now we should have everything we need to build the model

    m = mip.Model(sense=mip.MINIMIZE)
    m.threads = 16
    m.verbose = 1

    num_drones = 4

    # Define the variables for each drone
    x = {}
    for (i, j) in arcs:
        for d in range(num_drones):
            x[i, j, d] = m.add_var(var_type=mip.BINARY)
            # x[i, j, d] = 1 if arc (i, j) is used by drone d, 0 otherwise

    # Constraints
    grid_nodes = [i for i in V if i != 0]  # Except for the base node
    N_grid = len(grid_nodes)

    # Sub tour elimination variables
    u = {}
    for i in grid_nodes:
        for d in range(num_drones):
            u[i, d] = m.add_var(var_type=mip.CONTINUOUS, lb=0, ub=N_grid)


    # Number of drones leaving and entering a node must be 1
    for i in grid_nodes:
        # Incoming drones
        m += (mip.xsum(x[j, i, d] for (j, i2) in arcs if i2 == i for d in range(num_drones)) == 1)

        # Outgoing drones
        m += (mip.xsum(x[i, j, d] for (i2, j) in arcs if i2 == i for d in range(num_drones)) == 1)


    # Constraint for drones leaving and entering the base
    for d in range(num_drones):
        # Leave base exactly once
        m += (mip.xsum(x[0, j, d] for (i, j) in arcs if i == 0) == 1)

        # Return to base exactly once
        m += (mip.xsum(x[i, 0, d] for (i, j) in arcs if j == 0) == 1)

    # Flow conservation for each drone
    for d in range(num_drones):
        for i in grid_nodes:
            m += (mip.xsum(x[j, i, d] for (j, i2) in arcs if i2 == i) ==
                  mip.xsum(x[i, j, d] for (i2, j) in arcs if i2 == i))

    # MTZ sub tour elimination constraints
    for d in range(num_drones):
        for (i, j) in arcs:
            if i != 0 and j != 0:  # only grid nodes
                m += u[i, d] - u[j, d] + N_grid * x[i, j, d] <= N_grid - 1

    # MTZ: u[i,d] must be 0 if drone d does NOT visit node i
    for i in grid_nodes:
        for d in range(num_drones):
            incoming = mip.xsum(x[j, i, d] for (j, i2) in arcs if i2 == i)
            m += u[i, d] <= N_grid * incoming

    # Objective: minimize the maximum route length among drones
    T = [m.add_var(lb=0.0) for d in range(num_drones)]
    L = m.add_var(lb=0.0)

    for d in range(num_drones):
        m += (T[d] == mip.xsum(cost[i, j] * x[i, j, d] for (i, j) in arcs))
        m += T[d] <= L

    m.objective = L
    status = m.optimize()

    def reconstruct_and_print_routes(arcs, x, num_drones=4):
        for d in range(num_drones):
            # Build successor map: succ[i] = j if drone d travels i->j
            succ = {}
            for (i, j) in arcs:
                if x[i, j, d].x is not None and x[i, j, d].x > 0.5:
                    succ[i] = j

            # Reconstruct route
            route = [0]
            current = 0

            while True:
                if current not in succ:
                    # Safety break (should not happen in a valid solution)
                    break
                nxt = succ[current]
                route.append(nxt)
                current = nxt
                if current == 0:
                    break

            print(f"Drone {d + 1}: " + "-".join(str(v) for v in route))

    if status in (mip.OptimizationStatus.OPTIMAL,
                  mip.OptimizationStatus.FEASIBLE):
        print(f"\n")
        reconstruct_and_print_routes(arcs, x, num_drones)
    else:
        print("No feasible solution")


main()
