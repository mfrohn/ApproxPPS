import subprocess
import glob
import os
import re
import time
import heapq
import tempfile
import copy
from collections import defaultdict
from dataclasses import dataclass, field
from approxPPS import (_read_network, _complete_PPS_approximation)

VERBOSE = False
def dprint(*args, **kwargs):
    if VERBOSE: print(*args, **kwargs)

model_file = "bpps_model_LP.mos"

@dataclass
class BBNode:
    fixed_vars: dict = field(default_factory=dict)
    depth: int = 0
    LB: float = float("inf")
    UB: float = float("inf")

class PPSExactSolver:

    def __init__(self, network_file, character_file, network, p, exact=1):
        self.network_file = network_file
        self.character_file = character_file
        self.network = network
        self.p = p
        self.exact = exact
        self.results = {}
        self.best_solution = None
        self.W = float("inf")
        self.n_L = defaultdict(int)
        self.n_U = defaultdict(int)
        self.cost_L = defaultdict(float)
        self.cost_U = defaultdict(float)


    def solve_lp_relaxation(self, node=None):
        fix_file = "mosel_fix_input.txt"
        if node is not None and node.fixed_vars:
            #with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
            with open(fix_file, "w") as tmp:
                tmp.write(f"nfix: {len(node.fixed_vars)}\n")
                tmp.write("fix_v: [")
                tmp.write(" ".join(str(v) for v, s in node.fixed_vars))
                tmp.write("]\n")
                tmp.write("fix_s: [")
                tmp.write(" ".join(str(s) for v, s in node.fixed_vars))
                tmp.write("]\n")
                tmp.write("fix_val: [")
                tmp.write(" ".join(str(val) for val in node.fixed_vars.values()))
                tmp.write("]\n")
                tmp.close()
                fix_file = tmp.name

        start_LP = time.perf_counter()
        cmd = [
            "mosel", 
            model_file,
            f"InputFile1={self.character_file}",
            f"InputFile2={self.network_file}"
        ]
        if node is not None and node.fixed_vars:
            cmd.append(f"FixFile={fix_file}")

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        LP_time_elapsed = time.perf_counter() - start_LP
        if os.path.exists(fix_file):
            os.remove(fix_file)

        lp_match = re.search(r'LP_OPT\s*=\s*([\d\.]+)', result.stdout)
        if not lp_match:
            return None

        sol = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith("+++"):
                continue
            row = list(map(float, line.split()))
            sol.append(row)

        return {"LP_val": float(lp_match.group(1)),"LP_time": LP_time_elapsed,"LP_sol": sol}

    def is_binary_solution(self, solution):
        for i, row in enumerate(solution):
            for j, val in enumerate(row):
                if abs(val) < 1e-9:
                    continue
                if abs(val - 1) < 1e-9:
                    continue
                return False
        return True

    def translate_solution(self, solution):
        translated = defaultdict(int)
        for i, row in enumerate(solution):
            for j, val in enumerate(row):
                translated[(i,j)] = val
        return translated 


    def choose_branch_vertex(self, fractional_vars):
        scores = defaultdict(float)
        for var, val in fractional_vars:
            scores[var] += (self.cost_L[var] + self.cost_U[var])/(2*self.p+2.0)
        if not scores:
            return None
        return max(scores, key=scores.get)

    def solve(self):
        start_solver = time.perf_counter()
        root_lp_output = self.solve_lp_relaxation()
        root = BBNode()
        root.LB = root_lp_output["LP_val"]
        root.LB_sol = root_lp_output["LP_sol"]
        self.results["root_LB"] = root.LB
        self.results["root_LB_time"] = root_lp_output["LP_time"]
        root.UB, UB_time, root.UB_sol = _complete_PPS_approximation(network=copy.deepcopy(self.network),p=self.p)
        self.results["root_UB"] = root.UB
        self.results["root_UB_time"] = UB_time

        if not self.exact:
            self.results["OPT"] = -1
            self.results["OPT_time"] = -1
            self.results["Branches"] = -1
            return self.results

        queue = []
        counter = 0
        if root.LB < root.UB:
            heapq.heappush(queue,(counter,root))
        else:
            self.best_solution = root.UB_sol
            self.W = root.UB
        counter += 1

        while queue:
            _, node = heapq.heappop(queue)
            dprint(f"{self.W}, {node.LB}, {node.UB}, {counter}")

            if self.W < node.LB:
                continue

            if node.UB < self.W:
                self.best_solution = node.UB_sol
                dprint(f"upper bound yields best binary solution")
                self.W = node.UB

            if self.is_binary_solution(node.LB_sol) and node.LB < self.W:
                self.best_solution = self.translate_solution(node.LB_sol)
                dprint(f"lower bound yields best binary solution")
                self.W = node.LB
            elif node.LB < node.UB:
                fractional_vars = [
                    (v+1, s)
                    for v, row in enumerate(node.LB_sol)
                    for s, val in enumerate(row)
                    if 1e-9 < val < 1.0 - 1e-9
                ]
                new_nodes = {}
                for var in fractional_vars:
                    child = BBNode(
                        fixed_vars = node.fixed_vars.copy(),
                        depth = node.depth + 1
                    )
                    child.fixed_vars[var] = 1
                    lp_output = self.solve_lp_relaxation(child)
                    child.LB = lp_output["LP_val"]
                    child.LB_sol = lp_output["LP_sol"]
                    child.UB, _, child.UB_sol = _complete_PPS_approximation(network=copy.deepcopy(self.network),fixed_vars=child.fixed_vars,p=self.p)

                    self.cost_L[var] = (self.n_L[var] * self.cost_L[var] + (child.LB - node.LB)) / (self.n_L[var] + 1)
                    self.cost_U[var] = (self.n_U[var] * self.cost_U[var] + (node.UB - child.UB)) / (self.n_U[var] + 1)
                    self.n_L[var] += 1
                    self.n_U[var] += 1
                    new_nodes[var] = child

                branch_vertex = self.choose_branch_vertex(fractional_vars)
                if branch_vertex is None:
                    continue

                for var in fractional_vars:
                    if var[0] == branch_vertex:
                        heapq.heappush(queue,(counter,new_nodes[var]))
                        counter += 1

            if time.perf_counter() - start_solver >= 1800:
                break

        solver_time_elapsed = time.perf_counter() - start_solver
        self.results["OPT"] = self.W
        self.results["OPT_time"] = solver_time_elapsed
        self.results["Branches"] = counter-1
        return self.results


###########
# Example #
###########

#filepath = "data/Networks_50_5_1_1/n50_r5_d1_p1_22023network.graphml"
#N = _read_network(filepath)
#label_file = "data/Networks_50_5_1_1/n50_r5_d1_p1_22023mosel_labels"
#network_file = "data/Networks_50_5_1_1/n50_r5_d1_p1_22023mosel_network.txt"
#solver = PPSExactSolver(network_file,label_file,N,1)
#results = solver.solve()
#gap = (results['OPT'] - results['root_LB']) / results['OPT']
#print(f"+++{network_file} {results['root_LB']:.4f} {results['root_LB_time']:.2f} {results['root_UB']:.4f} {results['root_UB_time']:.2f} {gap:.4f} {results['OPT_time']:.4f} {results['Branches']}")




