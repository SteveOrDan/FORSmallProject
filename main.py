import sys

import mip
import csv
import os

import numpy as np
import math


def main():
    drones = 4  # number of drones
    k = 0  # number of nodes

    entry_threshold = 0  # y-coordinate threshold for entry points
    base = [0, 0, 0]  # starting point
    coords = {}  # coordinates are x (horizontal 1), y (vertical), z (horizontal 2)

    def that_other_connection_thingy(a, b):  # a,b are np.arrays
        res = 0
        for i in range(len(a)):
            if abs(a[i] - b[i]) > 0.5:
                res += 1
        return res

    def connected(i, j):  # a, b are points
        a = np.array(coords[i])
        b = np.array(coords[j])

        # Checks on entry points
        if i == 0 and j != 0 and b[1] <= entry_threshold:  # i == 0 corresponds to a being the base
            return True
        elif j == 0 and i != 0 and a[1] <= entry_threshold:  # j == 0 corresponds to b being the base
            return True
        elif i == j:
            return False

        d = np.linalg.norm(a - b)

        # Checks on normal grid points
        if d <= 4:
            return True
        elif (d <= 11) and (that_other_connection_thingy(a, b) <= 1):
            return True
        return False

    def compute_cost(a, b):  # a, b are np.arrays
        horizontal_dist = math.sqrt((a[0] - b[0]) ** 2 + (a[2] - b[2]) ** 2)
        vertical_dist = abs(a[1] - b[1])
        return max(horizontal_dist / 1.5, vertical_dist / (1 if a[1] < b[1] else 2))

    def read_csv(path):
        # Decide which building we have from the filename
        name = os.path.basename(path).lower()
        if "1" in name:
            ret_base = (0.0, -16.0, 0.0)
            ret_entry_y_threshold = -12.5
        else:
            ret_base = (0.0, -40.0, 0.0)
            ret_entry_y_threshold = -20.0

        ret_coords = {0: ret_base}  # node_id -> (x, y, z)

        # We ignore lines starting with '#' (comments).
        with open(path, newline="") as f:
            filtered = (line for line in f if not line.lstrip().startswith("#"))
            reader = csv.reader(filtered)
            header = next(reader, None)  # expect something like ["x", "y", "z"]

            for idx, row in enumerate(reader, start=1):
                if not row:  # skip empty lines
                    continue
                x, y, z = map(float, row[:3])
                ret_coords[idx] = (x, y, z)

        return ret_base, ret_coords, ret_entry_y_threshold  # ret_entry_points

    def build_graph():
        nodes = list(coords.keys())

        for i in nodes:
            for j in nodes:
                if i == j:
                    continue
                if connected(i, j):
                    a = np.array(coords[i])
                    b = np.array(coords[j])
                    c_ij = compute_cost(a, b)
                    edges.append((i, j))
                    cost[(i, j)] = c_ij

        return edges, cost

    def build_model(coords, edges, cost, drones=4):
        """
        Build the MILP model:

        - x[i,j,d] = 1 if drone d uses arc (i,j)
        - T[d] = total travel time of drone d
        - L = max_d T[d]
        """
        nodes = list(coords.keys())
        grid_nodes = [i for i in nodes if i != 0]
        N = len(grid_nodes)

        m = mip.Model(sense=mip.MINIMIZE)

        # Decision variables
        x = {}
        for (i, j) in edges:
            for d in range(drones):
                x[i, j, d] = m.add_var(var_type=mip.BINARY, name=f"x_{i}_{j}_{d}")

        # MTZ variables for subtour elimination (per drone & grid node)
        u = {}
        for i in grid_nodes:
            for d in range(drones):
                u[i, d] = m.add_var(
                    var_type=mip.CONTINUOUS, lb=1, ub=N, name=f"u_{i}_{d}"
                )

        # Route length for each drone + makespan
        T = [m.add_var(var_type=mip.CONTINUOUS, lb=0.0, name=f"T_{d}") for d in range(drones)]
        L = m.add_var(var_type=mip.CONTINUOUS, lb=0.0, name="L")

        # ------------------------------------------------------------------
        # Constraints
        # ------------------------------------------------------------------

        # 1) Each grid node has exactly one incoming and one outgoing edge (overall)
        for i in grid_nodes:
            # incoming
            m += (
                    mip.xsum(
                        x[j, i, d]
                        for (j, i2) in edges
                        if i2 == i
                        for d in range(drones)
                    )
                    == 1
            )
            # outgoing
            m += (
                    mip.xsum(
                        x[i, j, d]
                        for (i2, j) in edges
                        if i2 == i
                        for d in range(drones)
                    )
                    == 1
            )

        # 2) Each drone leaves base once and returns to base once
        for d in range(drones):
            # leave base
            m += (
                    mip.xsum(
                        x[0, j, d]
                        for (i, j) in edges
                        if i == 0
                    )
                    == 1
            )
            # return to base
            m += (
                    mip.xsum(
                        x[i, 0, d]
                        for (i, j) in edges
                        if j == 0
                    )
                    == 1
            )

        # 3) Flow conservation for each drone at each grid node
        for d in range(drones):
            for i in grid_nodes:
                m += (
                        mip.xsum(
                            x[j, i, d]
                            for (j, i2) in edges
                            if i2 == i
                        )
                        == mip.xsum(
                    x[i, j, d]
                    for (i2, j) in edges
                    if i2 == i
                )
                )

        # 4) MTZ subtour elimination: for each drone and pair of grid nodes
        for d in range(drones):
            for i in grid_nodes:
                for j in grid_nodes:
                    if i == j:
                        continue
                    if (i, j) in cost:  # arc exists and both are grid nodes
                        m += u[i, d] - u[j, d] + N * x[i, j, d] <= N - 1

        # 5) Route length T[d] and makespan L
        for d in range(drones):
            m += (
                    T[d]
                    == mip.xsum(
                cost[i, j] * x[i, j, d]
                for (i, j) in edges
            )
            )
            m += T[d] <= L

        # Objective: minimize makespan
        m.objective = L

        return m, x, T, L

    # ==================================================================================================================
    # Main execution starts here
    # ==================================================================================================================
    if len(sys.argv) != 2:
        print(f"Usage: python {os.path.basename(sys.argv[0])} <instance.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]

    base, coords, entry_threshold = read_csv(csv_path)

    edges = []
    cost = {}

    build_graph()


if __name__ == "__main__":
    main()
