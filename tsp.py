import csv
import os
import sys

import numpy as np
import math

import mip

import matplotlib.pyplot as plt


def plot_graph(coords, arcs, max_edges=None):
    """
    Plot the full directed graph (all nodes and all arcs).

    coords: dict node_id -> (x, y, z)
    arcs: list of (i, j)
    max_edges: if not None, limit number of arcs drawn (for big instances)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot nodes
    xs = []
    ys = []
    zs = []
    for i, (x, y, z) in coords.items():
        xs.append(x)
        ys.append(y)
        zs.append(z)
    ax.scatter(xs, ys, zs, s=10, label="nodes")

    # Plot arcs (directed edges)
    edge_count = 0
    for (i, j) in arcs:
        if max_edges is not None and edge_count >= max_edges:
            break
        x1, y1, z1 = coords[i]
        x2, y2, z2 = coords[j]
        ax.plot([x1, x2], [y1, y2], [z1, z2], alpha=0.2, linewidth=0.5)
        edge_count += 1

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Full graph (nodes + arcs)")
    plt.legend()
    plt.show()


def plot_solution(coords, arcs, x, num_drones=4):
    """
    Plot the solution: only arcs with x[i,j,d] == 1, colored by drone.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot all nodes
    xs = []
    ys = []
    zs = []
    for i, (x_coord, y_coord, z_coord) in coords.items():
        xs.append(x_coord)
        ys.append(y_coord)
        zs.append(z_coord)
    ax.scatter(xs, ys, zs, s=15, c="k", label="nodes")

    # Distinguish drones by color
    colors = ["r", "g", "b", "m", "c", "y"]
    for d in range(num_drones):
        color = colors[d % len(colors)]
        for (i, j) in arcs:
            val = x[i, j, d].x
            if val is not None and val > 0.5:
                x1, y1, z1 = coords[i]
                x2, y2, z2 = coords[j]
                ax.plot([x1, x2], [y1, y2], [z1, z2],
                        color=color, linewidth=2, alpha=0.9)
        ax.plot([], [], [], color=color, label=f"Drone {d+1}")  # legend handle

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Solution routes by drone")
    plt.legend()
    plt.show()


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
        # Skip comment lines starting with '#'
        filtered = (line for line in f if not line.lstrip().startswith("#"))
        reader = csv.reader(filtered)

        # Remove the first row with "x,y,z"
        _ = next(reader, None)

        # Deduplicate coordinates while reading
        seen = set()          # set of (x, y, z) we've already added
        idx = 1               # next node id (0 is the base)
        duplicates = []       # list of duplicate coords for info

        seen.add(base)

        for row in reader:
            if not row:
                continue
            try:
                x = float(row[0])
                y = float(row[1])
                z = float(row[2])
            except (ValueError, IndexError):
                # row malformed, skip it
                continue

            key = (x, y, z)

            if key in seen:
                duplicates.append(key)
                continue

            seen.add(key)
            coords[idx] = key
            idx += 1

    if duplicates:
        print(f"Found {len(duplicates)} duplicate coordinate rows, removed.")
    else:
        print("No duplicate coordinates found.")

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

        # Base–entry connections
        if vertex_a == 0 and vertex_b != 0:
            return b[1] <= entry_y_threshold  # base -> entry
        if vertex_b == 0 and vertex_a != 0:
            return a[1] <= entry_y_threshold  # entry -> base

        # No other base–grid edges (base nodes connection don't follow the following rules)
        if vertex_a == 0 or vertex_b == 0:
            return False

        if vertex_a == vertex_b:
            return False

        # Both are grid nodes: use connectivity rules
        d = np.linalg.norm(a - b)

        if d <= 4.0:
            return True
        elif (d <= 11.0) and (axis_past_threshold(a, b) <= 1):
            return True
        return False

    def compute_cost(i, j):
        a = coords[i]
        b = coords[j]

        horizontal_dist = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
        vertical_dist = abs(a[2] - b[2])

        return max((horizontal_dist / 1.5), (vertical_dist / (1 if a[2] < b[2] else 2)))

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
    print("Building and solving the model...")
    m = mip.Model(sense=mip.MINIMIZE)
    m.threads = 16
    m.verbose = 1
    print("Model built.")

    num_drones = 4

    print("Creating variables...")
    # y[d] = 1 if drone d is used (flies a route), 0 if it stays at base
    y = [m.add_var(var_type=mip.BINARY, name=f"y_{d}") for d in range(num_drones)]

    # Define the variables for each drone
    x = {}
    for (i, j) in arcs:
        for d in range(num_drones):
            x[i, j, d] = m.add_var(var_type=mip.BINARY)
            # x[i, j, d] = 1 if arc (i, j) is used by drone d, 0 otherwise

    # Constraints
    grid_nodes = [i for i in V if i != 0]  # Except for the base node
    N_grid = len(grid_nodes)

    print("Adding sub tour variables...")
    # Sub tour elimination variables
    u = {}
    for i in grid_nodes:
        for d in range(num_drones):
            u[i, d] = m.add_var(var_type=mip.CONTINUOUS, lb=0, ub=N_grid)

    print("Adding constraint number of drones leaving and entering a node must be 1")
    # Number of drones leaving and entering a node must be 1
    '''
    for i in grid_nodes:
        # Incoming drones
        m += (mip.xsum(x[j, i, d] for (j, i2) in arcs if i2 == i for d in range(num_drones)) == 1)

        # Outgoing drones
        m += (mip.xsum(x[i, j, d] for (i2, j) in arcs if i2 == i for d in range(num_drones)) == 1)
    '''
    # Each grid node must be visited exactly once (by exactly one drone)
    for i in grid_nodes:
        m += mip.xsum(
            x[j, i, d]
            for (j, i2) in arcs if i2 == i
            for d in range(num_drones)
        ) == 1
    print("Adding constraints for drones leaving and entering the base...")
    # Constraint for drones leaving and entering the base:
    # if y[d] = 1: one departure and one return
    # if y[d] = 0: no departure and no return
    for d in range(num_drones):
        # leave base
        m += (
                mip.xsum(x[0, j, d] for (i, j) in arcs if i == 0) == y[d]
        )
        # return to base
        m += (
                mip.xsum(x[i, 0, d] for (i, j) in arcs if j == 0) == y[d]
        )

    # If a drone is unused (y[d] = 0), it must not use any arc
    for (i, j) in arcs:
        for d in range(num_drones):
            m += x[i, j, d] <= y[d]

    print("Adding flow conservation constraints...")
    # Flow conservation for each drone
    for d in range(num_drones):
        for i in grid_nodes:
            m += (mip.xsum(x[j, i, d] for (j, i2) in arcs if i2 == i) ==
                  mip.xsum(x[i, j, d] for (i2, j) in arcs if i2 == i))

    print("Adding MTZ sub tour elimination constraints...")
    # MTZ sub tour elimination constraints
    for d in range(num_drones):
        for (i, j) in arcs:
            if i != 0 and j != 0:  # only grid nodes
                m += u[i, d] - u[j, d] + N_grid * x[i, j, d] <= N_grid - 1

    print("Adding MTZ linking constraints...")
    # MTZ: u[i,d] must be 0 if drone d does NOT visit node i
    for i in grid_nodes:
        for d in range(num_drones):
            incoming = mip.xsum(x[j, i, d] for (j, i2) in arcs if i2 == i)
            m += u[i, d] <= N_grid * incoming
            m += u[i, d] >= incoming

    # Objective: minimize the maximum route length among drones
    T = [m.add_var(lb=0.0) for d in range(num_drones)]
    L = m.add_var(lb=0.0)

    for d in range(num_drones):
        m += (T[d] == mip.xsum(cost[i, j] * x[i, j, d] for (i, j) in arcs))
        m += T[d] <= L

    print("Optimizing...")
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

    print("Status:", status)
    print("Best bound:", m.objective_bound)
    print("Best value:", m.objective_value)
    print("MIP gap:", m.gap)

    if status == mip.OptimizationStatus.OPTIMAL:
        print("Solution is proven optimal.")
    elif status == mip.OptimizationStatus.FEASIBLE:
        print("Feasible solution found, but not proven optimal.")

    if status in (mip.OptimizationStatus.OPTIMAL,
                  mip.OptimizationStatus.FEASIBLE):
        reconstruct_and_print_routes(arcs, x, num_drones)
        print(f"Reported makespan L = {L.x:.4f}")
        for d in range(num_drones):
            print(f"T[{d + 1}] = {T[d].x:.4f}")

        # Uncomment for plotting (Just do it for Building 4 to avoid explosions)
        # plot_solution(coords, arcs, x, num_drones)
    else:
        print("No feasible solution")


# Remember to delete useless prints and verbose model option for the submission
main()
