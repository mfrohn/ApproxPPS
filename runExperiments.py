import subprocess
import glob
import os
import sys
import re
import time
from approxPPS import (_read_network, _complete_PPS_approximation)
from solverPPS import PPSExactSolver

VERBOSE = True
def dprint(*args, **kwargs):
    if VERBOSE: print(*args, **kwargs)

script_dir = os.path.dirname(os.path.abspath(__file__))
p = int(sys.argv[1])
exact_solve = int(sys.argv[2])
network_dir = sys.argv[3]
files_path = os.path.join(script_dir, network_dir)
graph_files = sorted(glob.glob(os.path.join(files_path, "*network.graphml")))

results = {}
for graph_file in graph_files:
    label_file = os.path.join(files_path, os.path.basename(graph_file.replace("network.graphml", "mosel_labels.txt")))
    network_file = os.path.join(files_path, os.path.basename(graph_file.replace("network.graphml", "mosel_network.txt")))
    if not os.path.exists(label_file) or not os.path.exists(network_file):
        continue
    dprint(f"Solve PPS on {graph_file}...")
    N = _read_network(graph_file)
    solver = PPSExactSolver(network_file=network_file,character_file=label_file,network=N,p=p,exact=exact_solve)
    results[network_file] = solver.solve()


n_opt = 0
max_approx_ratio = 0
min_approx_ratio = 2
avg_approx_ratio = 0
avg_gap = 0
avg_approx_time = 0
avg_LP_time = 0
if exact_solve:
    dprint(f"Network OPT LP LP_time GAP ApproxPPS ApproxPPS_time ApproxPPS_ratio OPT_time Branches")
else:
    dprint(f"Network LP LP_time ApproxPPS ApproxPPS_time")
for network_file, vals in results.items():
    LB = vals.get("root_LB")
    OPT = vals.get("OPT")
    UB = vals.get("root_UB")
    gap = (OPT - LB) / OPT
    ratio = UB / OPT
    if exact_solve:
        print(f"+++{os.path.basename(network_file[3:])} {OPT:.4f} {LB:.4f} {vals['root_LB_time']:.2f} {gap:.4f} {UB:.4f} {vals['root_UB_time']:.2f} {ratio:.4f} {vals['OPT_time']:.4f} {vals['Branches']}")
    else:
        print(f"+++{os.path.basename(network_file[3:])} {LB:.4f} {vals['root_LB_time']:.2f} {UB:.4f} {vals['root_UB_time']:.2f}")
    if UB is not None:
        if OPT == UB:
            n_opt += 1
        else:
            avg_approx_ratio += ratio
            if ratio > max_approx_ratio:
                max_approx_ratio = ratio
            if ratio < min_approx_ratio:
                min_approx_ratio = ratio
    avg_gap += gap
    avg_LP_time += vals['root_LB_time']
    avg_approx_time += vals['root_UB_time']
avg_approx_ratio /= (len(results)-n_opt)
avg_gap /= len(results)
avg_LP_time /= len(results)
avg_approx_time /= len(results)
        
dprint(f"total number of solved instances: {len(results)}")
if exact_solve:
    dprint(f"ratio of instances with optimality certificate: {100*n_opt/len(results)}%")
    dprint(f"maximum approximation ratio: {max_approx_ratio}")
    dprint(f"minimum approximation ratio: {min_approx_ratio}")
    dprint(f"average approximation ratio: {avg_approx_ratio}")
dprint(f"average approximation time: {avg_approx_time}")
if exact_solve:
    dprint(f"average LP gap: {avg_gap}")
dprint(f"average LP time: {avg_LP_time}")





