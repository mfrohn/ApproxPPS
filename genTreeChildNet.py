import argparse
import random
import copy
import networkx as nx
import matplotlib.pyplot as plt
import ngesh  # v1.2.1

def printTree(T):
    roots = [n for n in T.nodes if T.in_degree(n) == 0]
    root = roots[0]
    def hierarchy_pos(G, root, width=1., vert_gap=0.2):
        def _hierarchy_pos(G, node, left, right, vert, pos):
            pos[node] = ((left+right)/2, vert)
            children = list(G.successors(node))
            if children:
                dx = (right-left)/len(children)
                nextx = left
                for child in children:
                    pos = _hierarchy_pos(G, child, nextx, nextx+dx, vert-vert_gap, pos)
                    nextx += dx
            return pos
        return _hierarchy_pos(G, root, 0, width, 1., {})
    pos = hierarchy_pos(T, root)
    plt.figure(figsize=(12, 8))
    nx.draw(T, pos, with_labels=True, node_size=700, node_color="lightgreen", arrows=True)
    plt.title("Binary Ngesh Tree as NetworkX DAG (Hierarchical Layout)")
    plt.show()

internal_counter = [1]

def generate_labeled_phylogenetic_network(n, r, maxRetDepth, p=1, treeChild=True, seed=1234):
    random.seed(seed)
    # Generate a binary Yule tree T using Ngesh
    tree = ngesh.gen_tree(birth=1.0,death=0.5,num_leaves=n,labels="enum",seed=seed)
    # Convert to NetworkX DAG with unique internal names
    T = nx.DiGraph()
    leaf_counter = [1]
    # label vertices
    def assign_names(node):
        if not node.children:
            node.name = f"L{leaf_counter[0]}"
            leaf_counter[0] += 1
        else:
            node.name = f"V{internal_counter[0]}"
            internal_counter[0] += 1
        for child in node.children:
            assign_names(child)
    assign_names(tree)
    # add edges to T
    def add_edges(node):
        for child in node.children:
            T.add_edge(node.name, child.name)
            add_edges(child)
    add_edges(tree)
    leaves = [v for v in T.nodes if T.out_degree(v) == 0]

    def reticulation_depth(N, node, ret_depth_cache=None):
        """
        Compute the maximum number of reticulations along any path from the root to `node`.
        """
        if ret_depth_cache is None:
            ret_depth_cache = {}
        if node in ret_depth_cache:
            return ret_depth_cache[node]
    
        preds = list(N.predecessors(node))
        if not preds:
            ret_depth_cache[node] = 0 # root
            return 0
    
        max_depth = max(reticulation_depth(N, p, ret_depth_cache) for p in preds)
        if N.in_degree(node) > 1:
            max_depth += 1
        ret_depth_cache[node] = max_depth
        return max_depth

    # Construct phylogenetic network N from T by adding r reticulations sequentially
    reticulations = set()
    N = T.copy()
    for _ in range(r):
        attempts = 0
        while True:
            attempts += 1
            if attempts > 10000:
                print("Could not add all reticulations while preserving the tree-child property.")
                break
            u, v = random.sample(list(N.nodes), 2)
            if u == v or N.has_edge(u, v):
                continue
            if treeChild and (N.in_degree(v)!=1 or N.in_degree(u)!=1):
                continue
            elif N.in_degree(v)==0 or N.in_degree(u)==0:
                continue
            if nx.has_path(N, v, u): # acyclicity
                continue
            parent_u = list(N.predecessors(u))[0]
            parent_v = list(N.predecessors(v))[0]
            if treeChild and (parent_u in reticulations or parent_v in reticulations):
                continue
            #sibling_u = next(c for c in N.successors(parent_u) if c != u)
            sibling_v = next(c for c in N.successors(parent_v) if c != v)
            if treeChild and (sibling_v in reticulations):
                continue
            # temporarily add the new reticulation
            x = f"V{internal_counter[0]}"
            y = f"V{internal_counter[0]+1}"
            internal_counter[0] += 2
            N.remove_edge(parent_u,u)
            N.add_edge(parent_u,x)
            N.add_edge(x,u)
            N.remove_edge(parent_v,v)
            N.add_edge(parent_v,y)
            N.add_edge(y,v)
            N.add_edge(x,y)

            descendants_y = nx.descendants(N,y)
            leaves_y = [w for w in descendants_y if N.out_degree(w) == 0]
            if any(reticulation_depth(N,leaf) > maxRetDepth for leaf in leaves_y):
                # maximum reticulation depth violated. Reverse reticulation addition
                N.remove_edge(x,y)
                N.remove_edge(y,v)
                N.remove_edge(parent_v,y)
                N.add_edge(parent_v,v)
                N.remove_edge(x,u)
                N.remove_edge(parent_u,x)
                N.add_edge(parent_u,u)
                N.remove_node(x)
                N.remove_node(y)
                internal_counter[0] -= 2
                continue

            reticulations.add(y)
            break

    # Generate leaf labeling
    labels = {leaf: random.randint(0, p) for leaf in leaves}

    return N, labels

# -------------------------------
# Export functions
# -------------------------------
def export_graphml(N, labels, path="network.graphml"):
    nx.set_node_attributes(N, labels, "character")
    nx.write_graphml(N, path)
    print(f"Network saved to {path}.")

def export_labels(labels, path="labels.txt"):
    with open(path, "w") as f:
        for leaf, lbl in labels.items():
            f.write(f"{leaf}\t{lbl}\n")
    print(f"Labels saved to {path}.")

def export_mosel_labels(leaves_labels,n,p, path="mosel_labels.txt"):
    leaves = list(leaves_labels.keys())
    # Relabel leaves: L1..Ln -> 1..n
    leaves_map = {leaf: i+1 for i, leaf in enumerate(leaves)}
    # Array of labels corresponding to 1..n
    C = [str(leaves_labels[leaf]) for leaf in leaves]
    with open(path, "w") as f:
        f.write(f"n: {n}\n")
        f.write(f"p: {p}\n")
        f.write(f"leaves: [ {' '.join(str(leaves_map[leaf]) for leaf in leaves)} ]\n")
        f.write(f"C: [ {' '.join(C)} ]\n")
    print(f"Leaves and labels saved to {path}.")

def export_mosel_network(N,n, path="mosel_network.txt"):
    U = []
    V = []
    for u, v in N.edges():
        if u[0] == 'V':
            U.append(int(u[1:])+n)
        if v[0] == 'V':
            V.append(int(v[1:])+n)
        if u[0] == 'L':
            U.append(int(u[1:]))
        if v[0] == 'L':
            V.append(int(v[1:]))
    m = len(N.edges())
    with open(path, "w") as f:
        f.write(f"m: {m}\n")  
        f.write(f"n_int: {internal_counter[0]-1}\n")                   
        f.write(f"E1: [ {' '.join(map(str,U))} ]\n")     
        f.write(f"E2: [ {' '.join(map(str,V))} ]\n")
    print(f"Edge list saved to {path}.")



def extended_newick_from_network(N, labels=None, path=None):
    """
    Convert a directed acyclic network N (networkx.DiGraph) to extended Newick.
    - labels: optional dict mapping leaf node -> label (string or int). If not given,
              leaf node names are used.
    - path: optional file path to write the resulting Newick string.
    Returns: extended Newick string.
    """
    M = N.copy()
    # sanity check: M must be a DAG
    if not nx.is_directed_acyclic_graph(M):
        raise ValueError("Input phylogenetic network must be a directed acyclic graph (DAG).")
    # sanity check: M has a unique root
    roots = [v for v in M.nodes if M.in_degree(v) == 0]
    if len(roots) != 1:
        raise ValueError("Input phylogenetic network is not rooted.")
    # prepare label lookup for leaves
    labels = labels or {}
    def leaf_repr(node):
        if node in labels:
            return str(labels[node])
        return str(node)

    # build extended Newick via recursion with hybrid id bookkeeping
    hybrid_ids = {}        # node -> "#Hk"
    hybrid_counter = [1] 

    def recurse(u):
        """Return extended-newick string for subtree rooted at u (u is a node in M)."""
        children = list(M.successors(u))
        if not children:
            return leaf_repr(u)
        if M.in_degree(u) > 1: # u is a reticulation
            # If we've already printed this hybrid node elsewhere, return its id only
            if u in hybrid_ids:
                return hybrid_ids[u]
            # Otherwise, assign hybrid id and print subtree once followed by id
            hid = f"#H{hybrid_counter[0]}"
            hybrid_ids[u] = hid
            hybrid_counter[0] += 1
            # For a reticulation we still print the child list, then the hybrid id
            inner = ",".join(recurse(c) for c in children)
            return "(" + inner + ")" + hid
        # u is an internal tree node or the root of M
        inner = ",".join(recurse(c) for c in children)
        # append node name to internal node label
        return "(" + inner + ")" + str(u)

    newick = recurse(roots[0])  # start at root
    newick_final = newick + ";"

    if path:
        with open(path, "w") as f:
            f.write(newick_final)
        print(f"Network saved to {path}.")

    return newick_final


# -------------------------------
#   main
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate phylogenetic networks with ngesh.")
    parser.add_argument("-n", type=int, required=True, help="Number of leaves")
    parser.add_argument("-r", type=int, required=True, help="Number of reticulations")
    parser.add_argument("-d", type=int, default=1, help="Maximum reticulation depth (default=1)")
    parser.add_argument("-p", type=int, default=1, help="Maximum label value (default=1)")
    parser.add_argument("-seed", type=int, default=1234, help="Random seed (default=1234)")
    parser.add_argument("-f", type=str, help="File directory")
    args = parser.parse_args()
    print(args)
    N, labels = generate_labeled_phylogenetic_network(n=args.n, r=args.r, p=args.p, maxRetDepth=args.d, treeChild=True, seed=args.seed)

    newick = extended_newick_from_network(N, labels)

    extended_newick_from_network(N, path=f"{args.f}/n{args.n}_r{args.r}_d{args.d}_p{args.p}_{args.seed}network.nwk")
    export_graphml(N,labels, path=f"{args.f}/n{args.n}_r{args.r}_d{args.d}_p{args.p}_{args.seed}network.graphml")
    export_mosel_network(N,args.n, path=f"{args.f}/n{args.n}_r{args.r}_d{args.d}_p{args.p}_{args.seed}mosel_network.txt")
    export_mosel_labels(labels,args.n,args.p,path=f"{args.f}/n{args.n}_r{args.r}_d{args.d}_p{args.p}_{args.seed}mosel_labels.txt")



