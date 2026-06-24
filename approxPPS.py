import random
import time
import copy
import networkx as nx
from collections import deque, defaultdict
from PhylogeneticNetwork import PhylogeneticNetwork
from PhylogeneticTree import PhylogeneticTree

VERBOSE = False
def dprint(*args, **kwargs):
    if VERBOSE: print(*args, **kwargs)

def _read_network(filepath: str) -> PhylogeneticNetwork:
    """Create a PhylogeneticNetwork from a .graphml file."""
    G = nx.read_graphml(filepath)
    if not isinstance(G, nx.DiGraph):
        print(f"Warning: The given file does not contain a DAG.")
        return None
    N = PhylogeneticNetwork()
    roots = [v_id  for v_id in G.nodes() if G.in_degree(v_id) == 0]
    if not roots or len(roots) >= 2:
        print("Warning: The DAG in the given file contains no unique root.")
        return None
    leaves = [v_id for v_id in G.nodes() if G.out_degree(v_id) == 0]
    tree_vertices = [v_id for v_id in G.nodes() if G.out_degree(v_id) == 2]
    ret_vertices = [v_id for v_id in G.nodes() if G.in_degree(v_id) == 2]
    # add vertices
    for v_id, attrs in G.nodes(data=True):
        if v_id == roots[0]:
            v_type = 'root'
        elif v_id in leaves:
            v_type = 'leaf'
        elif v_id in tree_vertices:
            v_type = 'tree'
        elif v_id in ret_vertices:
            v_type = 'reticulation'
        else:
            v_type = 'subdivision'
        input_C = attrs.get("character", None)
        N.add_vertex(v_id, v_type, input_C)
    # add edges
    for u_id, v_id in G.edges():
        N.G.add_edge(u_id, v_id)

    return N

def _run_SPS_approximation(network: PhylogeneticNetwork,full_reset=True,parental=False,final_call=False):
    network._initialize_vertices(full_reset)
    network._iterative_processing(parental)
    return network._finalize_extension_and_calculate_parsimony_score(parental,final_call)

def _complete_SPS_approximation(network: PhylogeneticNetwork):
    start = time.perf_counter()
    score, extension = None, None
    try:
        _run_SPS_approximation(network)
        extension, score = _run_SPS_approximation(network)
    except Exception as e:
        print(f"SPS approximation algorithm failed")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    algo_elapsed = time.perf_counter() - start
    return extension, score, algo_elapsed

def _get_subtree(network: PhylogeneticNetwork, v_id):
    subnetwork = network._get_reticulation_induced_subnetwork(v_id)
    subtree = PhylogeneticTree()
    dprint(f"subtree has root {subnetwork.root}")
    try:
        subtree.G = subnetwork.G.copy()
        subtree.root = subnetwork.root
        subtree.leaves = subnetwork.leaves.copy()
        subtree.k = max(subtree.G.nodes[leaf]['input_C'] for leaf in subtree.leaves)
        subtree._validate_tree()
    except ValueError as e:
        #print(f"Cannot convert: {e}")
        return None
    return subtree


def _solve_k_l_PPS(tree: PhylogeneticTree, l: int):
    dprint("solve a ( k,",l,")-PPS")
    #if tree.k>l: return None
    tree._initialize_vertices()
    for v_id in tree.G.nodes:
        if tree.G.nodes[v_id]['stateset']:
            tree.G.nodes[v_id]['stateset'] = [tree.G.nodes[v_id]['stateset']]
        else:
            tree.G.nodes[v_id]['stateset'] = []
    tree._iterative_processing_tree(l,True)
    dprint("done solving ( k,",l,")-PPS")
    return tree

def _calculate_reticulation_child_statesets(network: PhylogeneticNetwork):
    dprint("Calculate statessets for reticulation children inducing trees")
    for v_id in network.G.nodes:
        if network.G.nodes[v_id]['type'] != 'reticulation':
            continue
        child = next(network.G.successors(v_id))
        if network.valid_stateset(child):
            continue
        subtree = _get_subtree(network,child)
        if subtree is None: # subtree contains a vertex which is a reticulation vertex in network.G
            continue
        subtree = _solve_k_l_PPS(subtree,2)
        for v in subtree.G.nodes:
            dprint(v,subtree.G.nodes[v]['stateset'])
        dprint(f"run SPS approximation on subtree rooted in {child}")
        extension, score = _run_SPS_approximation(subtree.copy()) # only for root information
        for v in subtree.G.nodes:
            dprint(v,subtree.G.nodes[v]['stateset'],subtree.G.nodes[v]['score'])
        for w_id in subtree.G.nodes:
            if subtree.G.nodes[w_id]['type'] in ('tree','root','subdivision'):
                network.G.nodes[w_id]['status'] = subtree.G.nodes[w_id]['status']
                network.G.nodes[w_id]['stateset'] = subtree.G.nodes[w_id]['stateset']
            if subtree.G.nodes[w_id]['type'] == 'root':
                network.G.nodes[w_id]['extension'] = extension
                network.G.nodes[w_id]['extension_score'] = score
    dprint("Done with calculating statessets for ret children inducing trees.")
    for v in network.G.nodes:
        dprint(v,network.G.nodes[v]['stateset'],network.G.nodes[v]['score'])

def _score_dict(network: PhylogeneticNetwork, p: int):
    score_dict = {}
    for v_id, attrs in network.G.nodes(data=True):
        if isinstance(attrs['score'], (int, float)):
            score_dict[(v_id,attrs['score'])] = 1.0
            for s in set(range(0,p+1)) - set({attrs['score']}):
                score_dict[(v_id,s)] = 0.0
        elif isinstance(attrs['score'], set):
            for s in attrs['score']:
                score_dict[(v_id,s)] = 1.0
            for s in set(range(0,p+1)) - attrs['score']:
                score_dict[(v_id,s)] = 0.0
        else:
            raise ValueError(f"Unexpected score for {v_id}: {attrs['score']}")
    return score_dict

def _complete_PPS_approximation(network: PhylogeneticNetwork, fixed_vars: dict = None, p: int = 1):
    start = time.perf_counter()
    final_score = None
    try:
        dprint("Calculate a reticulation graph extension")
        network._initialize_vertices()
        if fixed_vars is not None:
            network._fix_variables(fixed_vars)
        _calculate_reticulation_child_statesets(network)
        network._construct_reticulation_graph_extension()
        dprint("Approximiate the PPS")
        pre_score = _run_SPS_approximation(network,False,True,False)
        dprint("pre score",pre_score)
        network._initialize_vertices_above_leaves()
        if fixed_vars is not None:
            network._fix_variables(fixed_vars)
        final_score = _run_SPS_approximation(network,False,True,True)
        dprint("final score",final_score)
    except Exception as e:
        print(f"PPS approximation algorithm failed")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    algo_elapsed = time.perf_counter() - start
    return final_score, algo_elapsed, _score_dict(network,p)


###########
# Example #
###########
        
#filepath = "data/Networks_50_5_1_1/n50_r5_d1_p1_22023network.graphml"
#N = _read_network(filepath)
#final_score, algo_elapsed, score_dict = _complete_PPS_approximation(network=N,p=1)
#print(filepath,"has parental parsimony score at most",final_score,f"({algo_elapsed * 1000:.2f} ms)")
#print(score_dict)

