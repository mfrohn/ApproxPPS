Step 1:

runNetworkGeneration creates rooted, binary, tree-child networks for chosen parameters by calling the method

genTreeChildNet.py -n[number of leaves] -r[number of reticulations] -d[max reticulation depth] -p[max label value] -seed[seed number] -f[files directory]

Output:
Random phylogenetic network satisfying the given parameters in mosel, graphml and newick format


Step 2:

For every set of networks, the following method calculates lower and upper bounds and optionally solve the PPS exactly:

runExperiments.py [max label value] [solve exactly (yes=1/no=0)] [network files directory]

The input for this method matches the output of genTreeChildNet.py.


solverPPS.py 		implements the exact solver
approxPPS.py 		implements the approximation algorithm ApproxPPS
bpps_model_LP.mos 	implements the LP relaxation for the binary network formulation of the PPS