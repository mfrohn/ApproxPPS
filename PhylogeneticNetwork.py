import random
import copy
import networkx as nx
from collections import deque
from typing import Tuple, Optional, Set, Deque

UNPROCESSED = "unprocessed"
PROCESSED = "processed"
VERBOSE = False

def dprint(*args, **kwargs):
    if VERBOSE: print(*args, **kwargs)

class PhylogeneticNetwork:
    """
    Phylogenetic network as a networkx.DiGraph with additional operations.

    Vertices are identified by their IDs with the following attributes:
      - type: 'leaf', 'tree', 'reticulation', 'root' or 'subdivision'
      - status: UNPROCESSED, PROCESSING or PROCESSED
      - stateset: set of states
      - score: final assigned state if it exists
      - input_C: input character if the vertex has type leaf
    """

    def __init__(self, graph = None):
        if graph is None:
            self.G = nx.DiGraph()
        else:
            self.G = graph
        self.root = None
        self.leaves = []

    def __deepcopy__(self, memo):
        new_network = PhylogeneticNetwork(copy.deepcopy(self.G, memo))
        new_network.root = copy.deepcopy(self.root, memo)
        new_network.leaves = copy.deepcopy(self.leaves, memo)
        return new_network

    def __repr__(self):
        lines = [f"Vertices of the phylogenetic network:"]
        for v_id, attrs in self.G.nodes(data=True):
            if self.G.nodes[v_id]['type'] != 'leaf':
                continue
            line = f"  {v_id}: type={attrs['type']}, status={attrs['status']}, score={attrs['score']}, stateset={attrs['stateset']}"
            lines.append(line)
        for v_id, attrs in self.G.nodes(data=True):
            if self.G.nodes[v_id]['type'] == 'leaf':
                continue
            line = f"  {v_id}: type={attrs['type']}, status={attrs['status']}, score={attrs['score']}, stateset={attrs['stateset']}"
            lines.append(line)
        for leaf in self.leaves:
            line = f" C({leaf}) = {self.G.nodes[leaf]['input_C']}"
            lines.append(line)
        return "\n".join(lines)

    def valid_stateset(self, v_id):
        if self.G.nodes[v_id]['type'] == 'leaf' or self.G.nodes[v_id]['status'] == PROCESSED:
            return True
        return False

    def add_vertex(self, v_id, v_type, input_C=None):
        """Adds a vertex to the network."""
        if v_id in self.G:
            raise ValueError(f"Vertex with ID {v_id} already exists.")
        if v_type not in ['leaf', 'tree', 'reticulation', 'root', 'subdivision']:
            raise ValueError(f"Invalid vertex type: {v_type}")
        self.G.add_node(v_id,type=v_type,status=UNPROCESSED,stateset=set(),score=None,input_C=input_C)
        if v_type == 'root':
            if self.root is not None:
                raise ValueError("Network cannot have multiple roots.")
            self.root = v_id
        elif v_type == 'leaf':
            self.leaves.append(v_id)

    def delete_vertex(self, v_id):
        """Deletes a vertex and its associated edges from the network."""
        if v_id not in self.G:
            raise ValueError(f"Vertex with ID {v_id} does not exist in the network.")
        v_type = self.G.nodes[v_id]['type']
        self.G.remove_node(v_id)
        if v_type == 'root':
            self.root = None
            print("Warning: Deleted root vertex. The network is now unrooted.")
        elif v_type == 'leaf':
            try:
                self.leaves.remove(v_id)
            except ValueError:
                pass
            #print(f"Warning: Deleted leaf vertex {v_id}.")

    def get_sibling(self, parent_id, child_id):
        """Return the sibling of child_id with respect to parent_id if it exists, else None."""
        children = list(self.G.successors(parent_id))
        if len(children) != 2:
            return None
        return children[0] if children[0] != child_id else (children[1] if children[1] != child_id else None)

    def _initialize_vertices(self, full_reset=True, character: dict = None):
        """Initializes internal vertices and assigns the character to the leaves if possible"""
        for v_id in self.G.nodes:
            self.G.nodes[v_id]['score'] = None
            if self.G.nodes[v_id]['type'] == 'leaf':
                if not character and self.G.nodes[v_id]['input_C']==None:
                    raise ValueError(f"Input character C missing for leaf {v_id}")
                if not character:
                    self.G.nodes[v_id]['stateset'] = {self.G.nodes[v_id]['input_C']}
                else:
                    self.G.nodes[v_id]['stateset'] = {character[v_id]}
                self.G.nodes[v_id]['status'] = PROCESSED
            elif full_reset or (self.G.nodes[v_id]['status'] == UNPROCESSED):
                self.G.nodes[v_id]['stateset'] = set()
                self.G.nodes[v_id]['status'] = UNPROCESSED  

    def _initialize_vertices_above_leaves(self):
        visited = set()
        stack = [self.root]
        while stack:
            v_id = stack.pop()
            visited.add(v_id)
            if v_id in visited or self.G.nodes[v_id]['type'] == 'leaf': 
                continue
            self.G.nodes[v_id]['score'] = None
            self.G.nodes[v_id]['stateset'] = set()
            self.G.nodes[v_id]['status'] = UNPROCESSED
            for child in list(self.G.successors(v_id)):
                if child not in visited:
                    stack.append(child)

    def _fix_variables(self, fixed_vars: dict = None):
        for var, val in fixed_vars.items():
            if not f"V{var[0]}" in self.G.nodes: 
                continue # can only be triggered if the current network is already a (parentally) displayed tree of the original network
            if not isinstance(self.G.nodes[f"V{var[0]}"]['stateset'],set):
                continue
            if val == 1:
                self.G.nodes[f"V{var[0]}"]['stateset'].add(var[1])
                self.G.nodes[f"V{var[0]}"]['status'] = UNPROCESSED

    def _process_tree_vertex(self, v_id) -> bool:
        """Processes a tree or root vertex or edge subdivision, updating its stateset based on children."""
        if self.G.nodes[v_id]['status'] != UNPROCESSED or self.G.nodes[v_id]['type'] not in ['tree', 'root', 'subdivision']:
            return False
        children = list(self.G.successors(v_id))
        if not children:
            if self.G.nodes[v_id]['type'] == 'root':
                self.G.nodes[v_id]['status'] = PROCESSED
                return True
            else:
                return False
        if self.G.nodes[v_id]['type'] == 'subdivision':
            self.G.nodes[v_id]['stateset'] = self.G.nodes[children[0]]['stateset']
            self.G.nodes[v_id]['status'] = PROCESSED
            return True
        if len(children) != 2:
            print(f"Warning: Tree/Root vertex {v_id} has {len(children)} children.")
            return False
        c1, c2 = children
        if self.G.nodes[c1]['status'] == UNPROCESSED or self.G.nodes[c2]['status'] == UNPROCESSED:
            return False
        # propagate statesets
        Cprime1 = self.G.nodes[c1]['stateset']
        Cprime2 = self.G.nodes[c2]['stateset']
        intersection = Cprime1 & Cprime2
        if intersection:
            dprint(f"Process tree vertex {v_id}. Take intersection to get stateset {intersection}")
            self.G.nodes[v_id]['stateset'] = intersection.copy()
            if self.G.nodes[v_id]['type'] == 'root' and len(self.G.nodes[v_id]['stateset']) > 1:
                # degree of freedom: the assigned state can be chosen randomly
                self.G.nodes[v_id]['stateset'] = {min(tuple(self.G.nodes[v_id]['stateset']))}
            if len(self.G.nodes[v_id]['stateset']) == 1:
                self._resolve(v_id)
        else:
            self.G.nodes[v_id]['stateset'] = Cprime1 | Cprime2
            dprint(f"Process tree vertex {v_id}. Take union to get stateset {self.G.nodes[v_id]['stateset']}")
        self.G.nodes[v_id]['status'] = PROCESSED
        return True

    def _resolve(self, v_id, resolve_plus=False) -> None:
        dprint(f"Resolve in {v_id} with stateset {self.G.nodes[v_id]['stateset']}")
        """Propagates a single character state assignment downwards from a resolved vertex."""
        stack = [v_id]
        while stack:
            vertex = stack.pop()
            for child in reversed(list(self.G.successors(vertex))):
                # skip fully resolved subnetworks
                if len(self.G.nodes[child]['stateset']) == 1:
                    continue
                #if self.G.has_node(f"{child}+"):
                #    continue 
                # skip all vertices in reticulation graphs except for the parents
                if self.G.nodes[child]['type'] == 'reticulation' or (f"{vertex}+" == child and not resolve_plus):
                    dprint(f"- stop at edge ({vertex},{child})")
                    continue
                if f"{vertex}+" == child:
                    resolve_plus=False
                dprint(f"- across edge ({vertex},{child})")
                if self.G.nodes[vertex]['stateset'] <= self.G.nodes[child]['stateset']:
                    self.G.nodes[child]['stateset'] = self.G.nodes[vertex]['stateset'].copy()
                elif len(self.G.nodes[child]['stateset']) >= 2:
                    # degree of freedom: the assigned state can be chosen randomly
                    self.G.nodes[child]['stateset'] = {min(tuple(self.G.nodes[child]['stateset']))}
                stack.append(child)

    def _resolve_score(self, root_id, parental) -> None:
        """Finalizes character assignments for all nodes based on the root's final state."""
        if self.G.nodes[root_id]['score'] is None:
            print(f"Error: Vertex {root_id} has no final character assignment.")
        stack = [root_id]
        while stack:
            vertex = stack.pop()
            for child in reversed(list(self.G.successors(vertex))):
                if parental:
                    if 'stateset' not in self.G.nodes[child]:
                        self.G.nodes[child]['stateset'] = {}
                    if len(self.G.nodes[child]['stateset']) >= 2:
                        self.G.nodes[child]['score'] = self.G.nodes[child]['stateset']
                    elif len(self.G.nodes[child]['stateset']) == 1:
                        self.G.nodes[child]['score'] = min(self.G.nodes[child]['stateset'])
                    else:
                        #print(f"Error: Child {child} of {vertex} has no character assignment.")
                        # this case can only be triggered by fixed statesets due to branching
                        parents = list(self.G.predecessors(child))
                        grandchild = list(self.G.successors(child))
                        if len(parents) == 2 and len(grandchild) == 1:
                            p1set = self.G.nodes[parents[0]]['stateset'] & self.G.nodes[grandchild[0]]['stateset']
                            p2set = self.G.nodes[parents[1]]['stateset'] & self.G.nodes[grandchild[0]]['stateset']
                            if len(p1set) <= len(p2set):
                                self.G.nodes[child]['stateset'] = p1set
                                self.G.nodes[child]['score'] = p1set
                            else:
                                self.G.nodes[child]['stateset'] = p2set
                                self.G.nodes[child]['score'] = p2set
                        elif len(grandchild) == 1:
                            self.G.nodes[child]['stateset'] = self.G.nodes[parents[0]]['stateset'] & self.G.nodes[grandchild[0]]['stateset']
                            self.G.nodes[child]['score'] = self.G.nodes[parents[0]]['stateset'] & self.G.nodes[grandchild[0]]['stateset']
                        else:
                            self.G.nodes[child]['stateset'] = self.G.nodes[parents[0]]['stateset']
                            self.G.nodes[child]['score'] = self.G.nodes[parents[0]]['stateset'] 
                    stack.append(child)
                    continue
                if len(self.G.nodes[child]['stateset']) == 0:
                    print(f"Error: Child {child} has no character assignment.")
                elif len(self.G.nodes[child]['stateset']) == 1:
                    self.G.nodes[child]['score'] = min(self.G.nodes[child]['stateset'])
                elif {self.G.nodes[vertex]['score']} <= self.G.nodes[child]['stateset']:
                    self.G.nodes[child]['score'] = self.G.nodes[vertex]['score']
                elif len(self.G.nodes[child]['stateset']) >= 2:
                    # degree of freedom: the assigned state can be chosen randomly
                    chosen_state = min(tuple(self.G.nodes[child]['stateset']))
                    self.G.nodes[child]['stateset'] = {chosen_state}
                    self.G.nodes[child]['score'] = chosen_state
                stack.append(child)

    def _get_reticulation_graph_vertices(self, ret_id, construct_ret_graph=False):
        """
        For a reticulation vertex return adjacent vertices and siblings 
        as an ordered set (parent1, parent2, ret_id, sibling1, sibling2, child)
        such that the sibling, and corresponding parent, with larger stateset-overlap with child comes first.
        """
        parents = list(self.G.predecessors(ret_id))
        children = list(self.G.successors(ret_id))
        if self.G.nodes[ret_id]['type'] != 'reticulation' or len(parents) != 2 or len(children) != 1:
            return None
        self.G.nodes[parents[0]]['ret_parent_match'] = parents[1]
        self.G.nodes[parents[1]]['ret_parent_match'] = parents[0]
        sibling1 = self.get_sibling(parents[0], ret_id)
        sibling2 = self.get_sibling(parents[1], ret_id)
        if sibling1 is None or sibling2 is None:
            print(f"Error: A parent of ({ret_id}) has only one child.")
            return None
        #if construct_ret_graph and (not self.G.nodes[children[0]]['stateset'] or not self.G.nodes[sibling1]['stateset'] or not self.G.nodes[sibling2]['stateset']):
        #    return None
        dprint(ret_id,parents,children,sibling1,sibling2)
        dprint(self.G.nodes[ret_id]['stateset'],self.G.nodes[children[0]]['stateset'],self.G.nodes[sibling1]['stateset'],self.G.nodes[sibling2]['stateset'])
        intersection1 = self.G.nodes[children[0]]['stateset'] & self.G.nodes[sibling1]['stateset']
        intersection2 = self.G.nodes[children[0]]['stateset'] & self.G.nodes[sibling2]['stateset']
        # degree of freedom for equality: first parent/sibling can be chosen randomly
        if len(intersection1) <= len(intersection2): # or (len(intersection1) == len(intersection2) and random.random() < 0.5):
            return parents[1], parents[0], ret_id, sibling2, sibling1, children[0]
        return parents[0], parents[1], ret_id, sibling1, sibling2, children[0]

    def _process_reticulation_graph(self, parent1, parent2, ret_id, sibling1, sibling2, child,parental) -> bool:
        """Processes a reticulation graph, potentially converting it to a tree."""
        if not (sibling1 and sibling2 and child):
            print(f"Error: Missing vertices for reticulation graph induced by {ret_id}. Cannot process.")
            return False
        if not (self.G.nodes[sibling1]['status'] == PROCESSED and self.G.nodes[sibling2]['status'] == PROCESSED and self.G.nodes[child]['status'] == PROCESSED):
            print(f"Error: Called _process_reticulation_graph for {ret_id} but prerequisites sibling1/sibling2/child not processed.")
            return False
        if not (self.G.nodes[parent1]['status'] == UNPROCESSED and self.G.nodes[parent2]['status'] == UNPROCESSED and self.G.nodes[ret_id]['status'] == UNPROCESSED):
            print(f"Warning: Processing reticulation graph induced by {ret_id}, but parent1/parent2/ret_vertex not all UNPROCESSED (Statuses: {self.G.nodes[parent1]['status']}/{self.G.nodes[parent2]['status']}/{self.G.nodes[ret_id]['status']}). Skipping.")
            return False

        stateset_parent1 = set()
        stateset_parent2 = None

        if sibling1 == parent2:
            # reticulation graph N_1
            sibling1, sibling2 = sibling2, sibling1
            parent1, parent2 = parent2, parent1

        dprint(f"Process reticulation graph: {sibling1} <-- {parent1} --> ret {ret_id} <-- {parent2} --> {sibling2}")
        dprint(f"Statesets {sibling1}:{self.G.nodes[sibling1]['stateset']} , ret_child {child}:{self.G.nodes[child]['stateset']} , {sibling2}:{self.G.nodes[sibling2]['stateset']}")

        if len(self.G.nodes[child]['stateset'] & self.G.nodes[sibling1]['stateset']) >= 1:
            # delete edge (parent2,ret_id) if child and sibling1 have a non-empty stateset overlap
            if not self._delete_ret_edge(parent1, parent2, ret_id, sibling2, child,parental):
                #print(f"Error: Failed to delete a reticulation edge for {ret_id}.")
                return False
            # resolve the statesets of child and sibling1
            if self.G.nodes[child]['stateset'] == self.G.nodes[sibling1]['stateset']:
                stateset_parent1 = self.G.nodes[child]['stateset'].copy()
            elif self.G.nodes[child]['stateset'] < self.G.nodes[sibling1]['stateset']:
                # shrink sibling1 to child's stateset and resolve downwards
                self.G.nodes[sibling1]['stateset'] = self.G.nodes[child]['stateset'].copy()
                stateset_parent1 = self.G.nodes[child]['stateset'].copy()
                self._resolve(sibling1)
            elif self.G.nodes[sibling1]['stateset'] < self.G.nodes[child]['stateset']:
                # shrink child to sibling1's stateset and resolve downwards
                self.G.nodes[child]['stateset'] = self.G.nodes[sibling1]['stateset'].copy()
                stateset_parent1 = self.G.nodes[sibling1]['stateset'].copy()
                self._resolve(child)
        else:
            # the stateset overlap between child and sibling1 is empty; build unions for both parents
            stateset_parent1 = self.G.nodes[sibling1]['stateset'].copy() | self.G.nodes[child]['stateset'].copy()
            stateset_parent2 = None
            if sibling1 != parent2: # no processing if edge (parent2,parent1) exists
                stateset_parent2 = self.G.nodes[sibling2]['stateset'].copy() | self.G.nodes[child]['stateset'].copy()

        self.G.nodes[parent1]['stateset'] = stateset_parent1
        self.G.nodes[parent1]['status'] = PROCESSED
        
        if stateset_parent2:
            # Reticulation graph remains unchanged
            self.G.nodes[ret_id]['stateset'] = self.G.nodes[parent1]['stateset'].copy()
            self.G.nodes[ret_id]['status'] = PROCESSED
            self.G.nodes[parent2]['stateset'] = stateset_parent2
            self.G.nodes[parent2]['status'] = PROCESSED

        if stateset_parent2:
            dprint(f"-> Statesets {parent1}:{self.G.nodes[parent1]['stateset']} , {parent2}:{self.G.nodes[parent2]['stateset']}")
        else:
            dprint(f"-> Stateset {parent1}:{self.G.nodes[parent1]['stateset']}")
        return True

    def _delete_ret_edge(self, parent1, parent2, ret_id, sibling2, child, parental) -> bool:
        """
        Remove the reticulation edge (parent2,ret_id) 
        and add edges (parent1,child) and (grandparent2,sibling2)
        """
        if parent1 not in self.G or parent2 not in self.G or ret_id not in self.G or sibling2 not in self.G or child not in self.G:
            #print(f"Error: parent ({parent1}), parent ({parent2}), reticulation ({ret_id}) or sibling ({sibling2}) or child ({child}) not present.")
            return True
        grandparent2 = list(self.G.predecessors(parent2))
        if len(grandparent2) != 1:
            #print(f"Error: parent ({parent2}) is not an internal tree vertex. Cannot process.")
            grandparent1 = list(self.G.predecessors(parent1))
            if len(grandparent1) == 1:
                self.skip[ret_id] = True # skip variable only for when fixed variable assignments from branching get stuck
                return True
            else:
                return False
        if parental and (f"{parent2}+" in self.G.nodes or (str(parent2).endswith('+') and parent2[:-1] in self.G.nodes)):
            self.G.nodes[parent2]['type'] = 'subdivision'
        else:
            dprint(f"-> delete vertex {parent2} and add edge ({grandparent2[0]},{sibling2})")
            self.delete_vertex(parent2)
            self.G.add_edge(grandparent2[0], sibling2)
        self.delete_vertex(ret_id)
        self.G.add_edge(parent1, child)
        dprint(f"-> delete vertex {ret_id} and add edge ({parent1},{child})")
        return True

    def _process_subdivided_tree_vertex(self, sub_id, vertex_id, child_id, sub_child_id) -> bool:
        """
        Assume (vertex_id,child_id), (vertex_id,sub_id) and (sub_id,sub_child_id) are edges of the network
        and sub_id is an edge subdivision.
        """
        if self.G.nodes[vertex_id]['status'] == PROCESSED or self.G.nodes[sub_id]['status'] == PROCESSED:
            print(f"Error: processed vertex {vertex_id} or {sub_id} already.")
            return False
        if sub_child_id is None:
            print(f"Error: missing a vertex.")
            return False
        elif self.G.nodes[child_id]['status'] == UNPROCESSED or self.G.nodes[sub_child_id]['status'] == UNPROCESSED:
            print(f"Error: unprocessed vertex {child_id} or {sub_child_id}.")
            return False
        union_stateset = self.G.nodes[child_id]['stateset'].copy() | self.G.nodes[sub_child_id]['stateset'].copy()
        self.G.nodes[vertex_id]['stateset'] = union_stateset.copy() 
        self.G.nodes[vertex_id]['status'] = PROCESSED
        self.G.nodes[sub_id]['stateset'] = union_stateset.copy()
        self.G.nodes[sub_id]['status'] = PROCESSED
        return True

    def _iterative_processing(self, parental=False) -> None:
        queue_tree:       deque = deque()
        id_queue_tree:    list[int] = []
        queue_ret:        deque = deque()
        id_queue_ret:     list[int] = []
        inconsistent_ret: list[int] = []

        def _add_tree_vertex(v):
            if self.G.nodes[v]['status'] != UNPROCESSED or v in id_queue_tree:
                return
            if (self.G.nodes[v]['type'] == 'root' and self.G.out_degree(v)==0) or (self.G.out_degree(v)>=1 and all(self.G.nodes[c]['status'] == PROCESSED for c in self.G.successors(v))):
                queue_tree.append(v); id_queue_tree.append(v)

        def _add_ret_vertex(v):
            dprint(f"add reticulation {v}")
            if self.G.nodes[v]['status'] != UNPROCESSED or v in id_queue_ret:
                return
            ret_graph_vertices = self._get_reticulation_graph_vertices(v)
            if not ret_graph_vertices:
                return
            p1, p2, r, s1, s2, c = ret_graph_vertices
            dprint(v,c,p1,p2)
            if (self.G.nodes[c]['status'] == PROCESSED and self.G.nodes[p1]['status'] == UNPROCESSED and self.G.nodes[p2]['status'] == UNPROCESSED):
                if (self.G.nodes[s1]['status'] == PROCESSED and self.G.nodes[s2]['status'] == PROCESSED):
                    queue_ret.append(ret_graph_vertices); id_queue_ret.append(r)
                    if r in inconsistent_ret: inconsistent_ret.remove(r)
                elif (self.G.nodes[s1]['status'] == PROCESSED or self.G.nodes[s2]['status'] == PROCESSED) and r not in inconsistent_ret:
                    inconsistent_ret.append(r)
            elif self.G.nodes[p1]['status'] == PROCESSED and self.G.nodes[p2]['status'] == PROCESSED:
                # this case can only be triggered by fixed statesets due to branching
                p1set = self.G.nodes[p1]['stateset'] & self.G.nodes[c]['stateset']
                p2set = self.G.nodes[p2]['stateset'] & self.G.nodes[c]['stateset']
                if len(p1set) >= len(p2set):
                    self.G.nodes[v]['stateset'] = p1set
                else:
                    self.G.nodes[v]['stateset'] = p2set
                self.G.nodes[v]['status'] = PROCESSED

        def _add_to_queues(V,addChildren=False):
            for v in V:
                if self.G.nodes[v]['type'] in ('tree', 'root', 'subdivision'):
                    _add_tree_vertex(v)
                elif self.G.nodes[v]['type'] == 'reticulation':
                    _add_ret_vertex(v)
                if addChildren:
                    for c in self.G.successors(v):
                        if self.G.nodes[c]['type'] == 'reticulation':
                            _add_ret_vertex(c)

        self.skip = {}
        for v in self.G.nodes:
            self.skip[v] = False # skip variable only for when fixed variable assignments from branching get stuck
        _add_to_queues(list(self.G.nodes))

        while True:
            dprint("\nCurrent tree-priority queue Q:")
            dprint(f"Priority 1: tree vertices {id_queue_tree}")
            dprint(f"Priority 2: two processed siblings of reticulation vertices {id_queue_ret}")
            dprint(f"Priority 3: one processed sibling of reticulation vertices {inconsistent_ret}\n")
            progress_made = False
            while queue_tree:
                v = queue_tree.popleft(); id_queue_tree.remove(v)
                if self._process_tree_vertex(v):
                    _add_to_queues(list(self.G.predecessors(v)),True)
                    progress_made = True

            if queue_ret:
                p1, p2, r, s1, s2, c = queue_ret.popleft(); id_queue_ret.remove(r)
                if self._process_reticulation_graph(p1, p2, r, s1, s2, c, parental):
                    for parent in (p1, p2):
                        if parent not in self.G or (self.G.in_degree(parent) == 1 and self.G.out_degree(parent) == 1):
                            continue
                        _add_to_queues(list(self.G.predecessors(parent)),True)
                    for sibling in (s1, s2):
                        for parent in list(self.G.predecessors(sibling)):
                            if self.G.nodes[parent]['status'] == PROCESSED: 
                                continue
                            _add_to_queues({parent},(self.G.in_degree(parent)!=1 or self.G.out_degree(parent)!=1))
                    progress_made = True

            if all(self.G.nodes[v]['status'] == PROCESSED for v in self.G.nodes):
                return

            if not progress_made:
                edge_subdivision_processed = False
                filter_ret = False
                to_remove = []
                to_check = []
                for v in list(inconsistent_ret):
                    ret_graph_vertices = self._get_reticulation_graph_vertices(v)
                    if not ret_graph_vertices:
                        to_remove.add(v)
                        continue
                    p1, p2, r, s1, s2, c = ret_graph_vertices
                    if s1 is None or s2 is None:
                        to_remove.add(r)
                        continue
                    # One sibling processed other not -> treat r as edge subdivision for processed sibling's parent
                    if not filter_ret and self.G.nodes[s1]['status'] == PROCESSED and self.G.nodes[s2]['status'] == UNPROCESSED and self.G.nodes[p1]['status'] == UNPROCESSED:
                        dprint(f"Process reticulation graph: {s1} <-- {p1} --> ret {r} <-- {p2} --> {s2}")
                        dprint(f"-> propagate statesets from {s1}:{self.G.nodes[s1]['stateset']} and {c}:{self.G.nodes[c]['stateset']} to {p1} without deleting ret {r}")
                        if self._process_subdivided_tree_vertex(r, p1, s1, c):
                            for grandparent in list(self.G.predecessors(p1)): to_check.append(grandparent)
                            edge_subdivision_processed = True
                            to_remove.append(r)
                    elif not filter_ret and self.G.nodes[s1]['status'] == UNPROCESSED and self.G.nodes[s2]['status'] == PROCESSED and self.G.nodes[p2]['status'] == UNPROCESSED:
                        dprint(f"Process reticulation graph: {s1} <-- {p1} --> ret {r} <-- {p2} --> {s2}")
                        dprint(f"-> propagate statesets from {s2}:{self.G.nodes[s2]['stateset']} and {c}:{self.G.nodes[c]['stateset']} to {p2} without deleting ret {r}")
                        if self._process_subdivided_tree_vertex(r, p2, s2, c):
                            for grandparent in list(self.G.predecessors(p2)): to_check.append(grandparent)
                            edge_subdivision_processed = True
                            to_remove.append(r)
                    elif self.G.nodes[s1]['status'] == PROCESSED and self.G.nodes[s2]['status'] == PROCESSED and self.G.nodes[c]['status'] == PROCESSED:
                        # reticulation was made consistent by processing another inconsistent reticulation
                        _add_to_queues({r})
                        filter_ret = True

                dprint("Remove verices ",to_remove)
                dprint("Queue candidates ",to_check)
                if edge_subdivision_processed:
                    inconsistent_ret = [r for r in inconsistent_ret if r not in to_remove]
                    # the order of to_check is fixed arbitrarly but affects the approximation quality
                    _add_to_queues(to_check,True)
                    progress_made = True
                
            if not progress_made:
                unprocessed = [v for v in self.G.nodes if self.G.nodes[v]['status'] == UNPROCESSED]
                # this case can only be triggered by fixed statesets due to branching
                dprint(f"Warning: Processing stuck. Unprocessed: {unprocessed}")
                for v in unprocessed:
                    for p in self.G.predecessors(v):
                        if self.G.nodes[p]['status'] == PROCESSED:
                            self.G.nodes[v]['stateset'] = self.G.nodes[p]['stateset']
                            self.G.nodes[v]['status'] = PROCESSED
                    for c in self.G.successors(v):
                        if self.G.nodes[c]['status'] == PROCESSED:
                            self.G.nodes[v]['stateset'] = self.G.nodes[c]['stateset']
                            self.G.nodes[v]['status'] = PROCESSED
                #return

    def _finalize_extension_and_calculate_parsimony_score(self,parental=False,final_call=False):
        """Finalizes character assignments and computes the total parsimony score."""
        if self.root is None:
            print("Error: Network has no root.")
            return None, None
        if self.G.nodes[self.root]['status'] == UNPROCESSED:
            print(f"Error: Root {self.root} was not processed. Status: {self.G.nodes[self.root]['status']}. Cannot finalize.")
            return None, None
        if not self.G.nodes[self.root]['stateset']:
            print(f"Warning: Root {self.root} stateset is empty after processing")
            return None, None

        dprint("\nFinalize statesets of size 2:")
        for v_id in list(self.G.nodes):
            if len(self.G.nodes[v_id]['stateset']) > 1:
                dprint(f"{v_id} : {self.G.nodes[v_id]['stateset']}")

        # Break up reticulation vertices into tree vertices based on minimum hamming distance
        deleting_ret = True
        to_remove = []
        while deleting_ret:
            deleting_ret = False
            vertices = [v_id for v_id in list(self.G.nodes) if v_id not in to_remove]
            for v_id in vertices:
                #resolve = True
                parents = list(self.G.predecessors(v_id))
                if self.G.nodes[v_id]['type'] != 'reticulation' and not any(self.G.has_edge(p,f"{p}+") for p in parents if f"{p}+" != v_id):
                    if len(self.G.nodes[v_id]['stateset']) == 1:
                        to_remove.append(v_id)
                        if any(len(self.G.nodes[c]['stateset'])>1 
                               and self.G.nodes[c]['type']!='reticulation' 
                               and not str(c).endswith('+') for c in list(self.G.successors(v_id))): 
                            dprint(f"resolve {v_id}")
                            self._resolve(v_id)
                    continue
                
                #dprint(f"{v_id} is reticulation vertex or induces a reticulation graph")
                # v_id is a reticulation vertex or induces a reticulation graph
                if any(len(self.G.nodes[p]['stateset']) > 1 for p in parents) and not str(v_id).endswith('+'):
                    #for p in parents:
                    #    dprint(self.G.nodes[p]['stateset'])
                    dprint(f"can't process {v_id} because parents {list(parents)} are unresolved")
                    continue
                if any(len(self.G.nodes[p]['stateset']) > 1 for p in parents):
                    dprint(f"reticulation graph resolution blocked for {v_id}")
                    if len(parents) == 2 and nx.has_path(self.G,parents[0],parents[1]):
                        self._resolve(parents[0][:-1],True)
                    if len(parents) == 2 and nx.has_path(self.G,parents[1],parents[0]):
                        self._resolve(parents[1][:-1],True)
                    continue
                # v_id is a reticulation vertex (or induces a reticulation graph) with resolved parent(s)
                if any(self.G.has_edge(p,f"{p}+") for p in parents if f"{p}+" != v_id):
                    # if (parent+,v_id+) is an edge: v_id induces an unresolved reticulation graph
                    # if (parent+,v_id+) is not an edge: v_id induces a partially unresolved reticulation graph with v_id unresolved
                    dprint(f"{v_id} is a {self.G.nodes[v_id]['type']} and induces a reticulation graph")
                    deleting_ret = self._finalize_reticulation_graph(parents[0],self.G.nodes[parents[0]]['ret_parent_match'],v_id)
                    to_remove.append(v_id)
                    if deleting_ret:
                        break
                    else:
                        continue
                else:
                    dprint(f"{v_id} is a {self.G.nodes[v_id]['type']} and does not induce a reticulation graph")
                
                # v_id is a reticulation vertex
                if len(parents) != 2:
                    dprint(f"{v_id} is a {self.G.nodes[v_id]['type']} with {len(parents)} parent(s)")
                    to_remove.append(v_id)
                    continue
                child_stateset = self.G.nodes[next(self.G.successors(v_id))]['stateset']
                
                dprint(f"Consider the Hamming distance between {next(self.G.successors(v_id))}:{self.G.nodes[next(self.G.successors(v_id))]['stateset']} and {parents[0]}:{self.G.nodes[parents[0]]['stateset']} , {parents[1]}:{self.G.nodes[parents[1]]['stateset']}")
                d_H, p_state, c_state = zip(*[self.hamming_dist(self.G.nodes[p]['stateset'],child_stateset) for p in parents])
                # if d_H[0] < d_H[1] delete edge (parents[1],child), otherwise (parents[0],child)
                # degree of freedom for equality: edge to be deleted can be chosen randomly
                if (d_H[0] < d_H[1] and parents[1] !=self.root) or (d_H[0] >= d_H[1] and parents[0] == self.root) or (self.skip[v_id] and d_H[0] != d_H[1]): # skip variable only for when fixed variable assignments from branching get stuck
                    deleting_ret = self._delete_ret_edge(parents[0], parents[1], v_id, self.get_sibling(parents[1], v_id), next(self.G.successors(v_id),None),parental)
                    self.G.nodes[parents[0]]['stateset'] = {p_state[0]}
                elif d_H[0] != d_H[1]:
                    deleting_ret = self._delete_ret_edge(parents[1], parents[0], v_id, self.get_sibling(parents[0], v_id), next(self.G.successors(v_id),None),parental)
                    self.G.nodes[parents[1]]['stateset'] = {p_state[1]}
                else:
                    r = random.randint(0,1)
                    deleting_ret = self._delete_ret_edge(parents[r], parents[1-r], v_id, self.get_sibling(parents[1-r], v_id), next(self.G.successors(v_id),None),parental)
                    self.G.nodes[parents[r]]['stateset'] = {p_state[r]}

                if deleting_ret:
                    break
            # loop until no reticulation left or no deletions possible

        dprint("Terminate with statesets of size 2:")
        for v_id in list(self.G.nodes):
            if len(self.G.nodes[v_id]['stateset']) > 1:
                dprint(f"{v_id} : {self.G.nodes[v_id]['stateset']}")

        if parental: 
            #self._recover_parentally_displayed_tree()
            if final_call: self._recover_final_tree()

        if len(self.G.nodes[self.root]['stateset']) >= 2:
            #self.G.nodes[self.root]['stateset'] = {random.choice(tuple(self.G.nodes[self.root]['stateset']))}
            for child in self.G.successors(self.root):
                if child == 'reticulation':
                    self.G.nodes[self.root]['stateset'] &= self.G.nodes[child]['stateset']
            self.G.nodes[self.root]['stateset'] = {min(tuple(self.G.nodes[self.root]['stateset']))}

        self.G.nodes[self.root]['score'] = min(self.G.nodes[self.root]['stateset'])     
        self._resolve_score(self.root,parental)
   
        for v_id in list(self.G.nodes):
            if not 'score' in self.G.nodes[v_id]:
                print(f"Warning: Vertex {v_id} has no final character assignment.")
                return None, None

        if not parental: 
            final_extension = {v_id: self.G.nodes[v_id]['score'] for v_id in self.G.nodes}

            parsimony_score = 0
            processed_edges = set()
            for u_id, v_id in self.G.edges():
                if (u_id, v_id) in processed_edges:
                    print(f"Error: Edge ({u_id},{v_id}) appears twice in the solution.")
                    continue
                parsimony_score += int(self.G.nodes[u_id]['score'] != self.G.nodes[v_id]['score'])
                processed_edges.add((u_id,v_id))

            return final_extension, parsimony_score

        if parental:
            dprint("calculate parental parsimony score")
            parental_parsimony_score = 0
            processed_edges = set()
            processed_parents = {}
            for u_id, v_id in self.G.edges():
                if (u_id, v_id) in processed_edges:
                    print(f"Error: Edge ({u_id},{v_id}) appears twice in the solution.")
                    continue
                if isinstance(self.G.nodes[v_id]['score'],set):
                    dprint("processing",self.G.nodes[v_id]['score'],"for",v_id," and parent ",u_id," with stateset ",self.G.nodes[u_id]['score'])
                    if self.G.nodes[u_id]['score'] != self.G.nodes[v_id]['score'] and self.G.nodes[u_id]['score'] not in self.G.nodes[v_id]['score']:
                        dprint(self.G.nodes[u_id]['score']," is not in ",self.G.nodes[v_id]['score'])
                        parental_parsimony_score += 1
                    else:
                        if v_id in processed_parents and self.G.nodes[processed_parents[v_id]]['score'] == self.G.nodes[u_id]['score']:
                            # now we have identified two parents of v_id at hamming distance 0
                            # -> one of the two contributes a cut edge
                            dprint(f"score for edge ({u_id},{v_id}) = 1 because its the second parent at distance zero.")
                            parental_parsimony_score += 1
                        else:
                            dprint(f"score for edge ({u_id},{v_id}) = 0 because its the first parent at distance zero.")
                        processed_parents[v_id] = u_id
                elif isinstance(self.G.nodes[u_id]['score'],set):
                    dprint("processing",self.G.nodes[u_id]['score'],"for",u_id," and child ",v_id," with stateset ",self.G.nodes[v_id]['score'])
                    if self.G.nodes[v_id]['score'] != self.G.nodes[u_id]['score'] and self.G.nodes[v_id]['score'] not in self.G.nodes[u_id]['score']:
                        dprint(self.G.nodes[v_id]['score']," is not in ",self.G.nodes[u_id]['score'])
                        parental_parsimony_score += 1
                else:
                    if int(self.G.nodes[u_id]['score'] != self.G.nodes[v_id]['score']) > 0: 
                        dprint(f"score for edge ({u_id},{v_id}) =",int(self.G.nodes[u_id]['score'] != self.G.nodes[v_id]['score']))
                    parental_parsimony_score += int(self.G.nodes[u_id]['score'] != self.G.nodes[v_id]['score'])
                processed_edges.add((u_id,v_id))
            return parental_parsimony_score



    def _get_reticulation_induced_subnetwork(self,v_id):
        # collect all vertex IDs reachable from v_id
        visited = set()
        stack = [v_id]
        while stack:
            vertex = stack.pop()
            if vertex in visited: continue
            visited.add(vertex)
            for child in self.G.successors(vertex):
                stack.append(child)

        subnetwork = PhylogeneticNetwork()
        # add vertices preserving their types where possible
        for w_id in visited:
            w_type = self.G.nodes[w_id]['type']
            if w_id == v_id:
                subnetwork.add_vertex(w_id, 'root')
            elif w_type in ('tree', 'root'):
                subnetwork.add_vertex(w_id, w_type)
            elif w_type == 'reticulation':
                # include reticulation only if all parents are visited
                parents = self.G.predecessors(w_id)
                parents_visited = all(p in visited for p in parents)
                subnetwork.add_vertex(w_id, 'reticulation' if parents_visited else 'subdivision')
            elif w_type == 'leaf': 
                subnetwork.add_vertex(w_id, w_type, self.G.nodes[w_id]['input_C'])
            else:
                subnetwork.add_vertex(w_id, w_type)
        # add edges among included vertices according to network self.G
        for u_id, w_id in self.G.edges():
            if u_id in subnetwork.G and w_id in subnetwork.G:
                subnetwork.G.add_edge(u_id, w_id)
        return subnetwork

    def _get_reticulation_induced_subnetworks(self):
        """Return a queue of subnetworks induced by each reticulation vertex."""
        queue_subnetworks:    deque[PhylogeneticNetwork] = deque()
        id_queue_subnetworks: set[int] = set()

        for v_id in list(self.G.nodes):
            if self.G.nodes[v_id]['type'] == 'reticulation':
                queue_subnetworks.append(self._get_reticulation_induced_subnetwork(v_id))
                id_queue_subnetworks.add(v_id)
        return queue_subnetworks, id_queue_subnetworks

    def _construct_reticulation_graph_extension(self):
        """Transforms the network into its reticulation graph extension
           Assumption: all children of reticulation vertices are processed"""
        dprint("Construct reticulation graph extension...")
        for v_id in list(self.G.nodes):
            if self.G.nodes[v_id]['type'] != 'reticulation':
                continue
            ret_graph_vertices = self._get_reticulation_graph_vertices(v_id,True)
            if not ret_graph_vertices:
                continue
            p1, p2, r, s1, s2, c = ret_graph_vertices
            if len(self.G.nodes[c]['stateset']) <= 1:
                continue
            if len(self.G.nodes[c]['stateset']) != 2:
                print(f"Warning:",v_id," does not have two states.")
            self.add_vertex(c+"-",'leaf')
            self.G.nodes[c]['initial_children'] = list(self.G.successors(c))
            for child in list(self.G.successors(c)):
                self.G.remove_edge(c,child)
            self.G.nodes[c]['type'] = 'leaf'
            dprint(f"Construct reticulation graph extension for v_r = {c}")
            # split the stateset of c into the statesets of c and c^+
            self.G.nodes[c+"-"]['input_C'] = list(self.G.nodes[c]['stateset'])[0]
            self.G.nodes[c]['input_C'] = list(self.G.nodes[c]['stateset'])[1]
            if s1 == p2:
                # the reticulation graph contains triangle (p1,p2), (p1,r), (p2,r)
                self.add_vertex(r+"+",'tree')
                self.G.remove_edge(p1,p2)
                self.G.add_edge(p1,r+"+")
                self.G.add_edge(r+"+",p2)
            elif s2 == p1:
                # the reticulation graph contains triangle (p2,p1), (p2,r), (p1,r)
                self.add_vertex(r+"+",'tree')
                self.G.remove_edge(p2,p1)
                self.G.add_edge(p2,r+"+")
                self.G.add_edge(r+"+",p1)
            else: # the reticulation graph contains no triangles
                self.add_vertex(r+"+",'reticulation')
                # subdivide edge (p1,s1) by p1^+ and insert edge (p1^+,r^+)
                self.G.remove_edge(p1,s1)
                self.add_vertex(p1+"+",'tree')
                self.G.add_edge(p1,p1+"+")
                self.G.add_edge(p1+"+",s1)
                self.G.add_edge(p1+"+",r+"+")
                # subdivide edge (p2,s2) by p2^+ and insert edge (p2^+,r^+)
                self.G.remove_edge(p2,s2)
                self.add_vertex(p2+"+",'tree')
                self.G.add_edge(p2,p2+"+")
                self.G.add_edge(p2+"+",s2)
                self.G.add_edge(p2+"+",r+"+")
            # insert edge (r^+,c^+)
            self.G.add_edge(r+"+",c+"-")
        dprint("Done constructing reticulation graph extension.")

    def hamming_dist(self,stateset1,stateset2):
        if stateset1 and stateset2:
            opt_states = min(
                ((state1, state2) for state1 in stateset1 for state2 in stateset2),
                key=lambda pair: int(pair[0] != pair[1])
            )
            return int(opt_states[0] != opt_states[1]), opt_states[0], opt_states[1]
        else:
            if stateset1 and not stateset2:
                return int(list(stateset1)[0] != 0), list(stateset1)[0], 0
            elif not stateset1 and stateset2:
                return int(list(stateset2)[0] != 0), 0, list(stateset2)[0]
            else:
                return 0, 0, 0

    def _finalize_reticulation_graph(self, parent1, parent2, v_id):
        if not self.G.has_edge(parent1,f"{parent1}+"):
            parent1, parent2 = parent2, parent1
        v_id_plus = next((w_id for w_id in self.G.successors(f"{parent1}+") if str(w_id).endswith('+')), None)
        if v_id_plus == None:
            dprint("-> reticulation graph is parentally displayed")
            return False
        dprint(f"-> consider edges ({parent1},{v_id}) and ({parent1}+,{v_id_plus}) to resolve reticulation graph")
        child_v_id = next(self.G.successors(v_id),None)
        child_v_id_plus = next(self.G.successors(v_id_plus),None)
        #dprint(self.G.nodes[v_id]['type'],self.G.nodes[v_id_plus]['type'])
        #if child_v_id != None:
        #    dprint(f"{child_v_id}:",self.G.nodes[child_v_id]['type'])
        #if child_v_id_plus != None:
        #    dprint(f"{child_v_id_plus}:",self.G.nodes[child_v_id_plus]['type'])
        if self.G.nodes[v_id]['type'] == 'reticulation' and self.G.nodes[v_id_plus]['type'] == 'reticulation': 
            dprint(f"-> {v_id} induces a fully unresolved reticulation graph")
            d_H11, p_state11, c_state11 = self.hamming_dist(self.G.nodes[parent1]['stateset'],self.G.nodes[child_v_id]['stateset'])
            d_H12, p_state12, c_state12 = self.hamming_dist(self.G.nodes[parent2]['stateset'],self.G.nodes[child_v_id]['stateset'])
            d_H21, p_state21, c_state21 = self.hamming_dist(self.G.nodes[f"{parent1}+"]['stateset'],self.G.nodes[child_v_id_plus]['stateset'])
            d_H22, p_state22, c_state22 = self.hamming_dist(self.G.nodes[f"{parent2}+"]['stateset'],self.G.nodes[child_v_id_plus]['stateset'])      
            if d_H11+d_H22+int(p_state11==p_state22) <= d_H12+d_H21+int(p_state12==p_state21):
                parent1, parent2 = parent2, parent1
                #self.G.nodes[child_v_id]['stateset'] = {c_state11} 
                #self.G.nodes[child_v_id_plus]['stateset'] = {c_state22}
            #else:
            #    p_state11, p_state12, p_state21, p_state22 = p_state12, p_state11, p_state22, p_state21
                #self.G.nodes[child_v_id]['stateset'] = {c_state12} 
                #self.G.nodes[child_v_id_plus]['stateset'] = {c_state21}
            #if p_state11 == p_state22:
                # a displayed and parentally displayed tree have the same parsimony score in this reticultion graph
                # -> taking the displayed tree offers more degrees of freedom for resolving statesets
                # however taking the displayed tree increases the total parsimony by at least one by construction
            # delete reticulation edge (parent1,v_id)
            self._delete_ret_edge(parent2, parent1, v_id, self.get_sibling(parent1, v_id), next(self.G.successors(v_id),None), True)
            # delete reticulation edge (parent2+,v_id+)
            self._delete_ret_edge(f"{parent1}+", f"{parent2}+", v_id_plus, self.get_sibling(f"{parent2}+", v_id_plus), next(self.G.successors(v_id_plus),None), True)
            # only _resolve(v_id,True) propagtes across the first occurrence of an edge of the form (v_id,v_id+)
            self._resolve(parent1,True)
            self._resolve(parent2,True)
            return True
        if self.G.nodes[v_id]['type'] == 'reticulation' and self.G.nodes[v_id_plus]['type'] != 'reticulation':
            d_H11, p_state11, c_state11 = self.hamming_dist(self.G.nodes[parent1]['stateset'],self.G.nodes[child_v_id]['stateset'])
            d_H12, p_state12, c_state12 = self.hamming_dist(self.G.nodes[parent2]['stateset'],self.G.nodes[child_v_id]['stateset'])
            d_H21, p_state21, c_state21 = self.hamming_dist(self.G.nodes[f"{parent1}+"]['stateset'],self.G.nodes[v_id_plus]['stateset'])
            d_H22, p_state22, c_state22 = self.hamming_dist(self.G.nodes[f"{parent2}+"]['stateset'],self.G.nodes[v_id_plus]['stateset'])      
            dprint(f"-> {v_id} induces a partially unresolved reticulation graph")
            if d_H11+d_H22+int(p_state11==p_state22) <= d_H12+d_H21+int(p_state12==p_state21):
                parent1, parent2 = parent2, parent1
            #    self.G.nodes[child_v_id]['stateset'] = {c_state11} 
            #    self.G.nodes[v_id_plus]['stateset'] = {c_state22}
            #else:
            #    self.G.nodes[child_v_id]['stateset'] = {c_state12} 
            #    self.G.nodes[v_id_plus]['stateset'] = {c_state21}
            # delete reticulation edge (parent1,v_id)
            self._delete_ret_edge(parent2, parent1, v_id, self.get_sibling(parent1, v_id), next(self.G.successors(v_id),None), True)
            if self.G.has_edge(f"{parent2}+",v_id_plus):
                # delete edge (parent2+,v_id+) and add edge (parent1+,v_id+)
                self.G.remove_edge(f"{parent2}+",v_id_plus)
                self.G.add_edge(f"{parent1}+",v_id_plus)
            self._resolve(parent1,True)
            self._resolve(parent2,True)
            return True
        if self.G.nodes[v_id]['type'] != 'reticulation' and self.G.nodes[v_id_plus]['type'] == 'reticulation':
            d_H11, p_state11, c_state11 = self.hamming_dist(self.G.nodes[parent1]['stateset'],self.G.nodes[v_id]['stateset'])
            d_H12, p_state12, c_state12 = self.hamming_dist(self.G.nodes[parent2]['stateset'],self.G.nodes[v_id]['stateset'])
            d_H21, p_state21, c_state21 = self.hamming_dist(self.G.nodes[f"{parent1}+"]['stateset'],self.G.nodes[child_v_id_plus]['stateset'])
            d_H22, p_state22, c_state22 = self.hamming_dist(self.G.nodes[f"{parent2}+"]['stateset'],self.G.nodes[child_v_id_plus]['stateset'])      
            dprint(f"-> {v_id} induces a partially unresolved reticulation graph")
            if d_H11+d_H22+int(p_state11==p_state22) <= d_H12+d_H21+int(p_state12==p_state21):
                parent1, parent2 = parent2, parent1
            #    self.G.nodes[v_id]['stateset'] = {c_state11} 
            #    self.G.nodes[child_v_id_plus]['stateset'] = {c_state22}
            #else:
            #    self.G.nodes[v_id]['stateset'] = {c_state12} 
            #    self.G.nodes[child_v_id_plus]['stateset'] = {c_state21}
            if self.G.has_edge(parent1,v_id):
                # delete edge (parent1,v_id) and add edge (parent2,v_id)
                self.G.remove_edge(parent1,v_id)
                self.G.add_edge(parent2,v_id)
            # delete reticulation edge (parent2+,v_id+)
            self._delete_ret_edge(f"{parent1}+", f"{parent2}+", v_id_plus, self.get_sibling(f"{parent2}+", v_id_plus), next(self.G.successors(v_id_plus),None), True)
            self._resolve(parent1,True)
            self._resolve(parent2,True)
            return True
        if self.G.nodes[v_id]['type'] != 'reticulation' and self.G.nodes[v_id_plus]['type'] != 'reticulation':
            # this case is only included for completion. It is never called by the complete approximation algorithm
            dprint(f"stateset of {parent1} = {self.G.nodes[parent1]['stateset']}, {v_id} = {self.G.nodes[v_id]['stateset']}")
            dprint(f"stateset of {parent2} = {self.G.nodes[parent2]['stateset']}, {v_id} = {self.G.nodes[v_id]['stateset']}")
            d_H11, p_state11, c_state11 = self.hamming_dist(self.G.nodes[parent1]['stateset'],self.G.nodes[child_v_id]['stateset'])
            d_H21, p_state21, c_state21 = self.hamming_dist(self.G.nodes[f"{parent1}+"]['stateset'],self.G.nodes[child_v_id_plus]['stateset'])
            if d_H11 > d_H21:
                dprint(f"delete ({parent1},{v_id}) and add ({parent2},{v_id})")
                self.G.remove_edge(parent1,v_id)
                self.G.add_edge(parent2,v_id)
            elif d_H11 < d_H21:
                dprint(f"delete ({parent1}+,{v_id_plus}) and add ({parent2}+,{v_id_plus})")
                self.G.remove_edge(f"{parent1}+",v_id_plus)     
                self.G.add_edge(f"{parent2}+",v_id_plus)
            return False



    def _recover_parentally_displayed_tree(self):
        dprint("recover parentally displayed trees")
        """Transforms the network into a parentally displayed tree of the original network
           Assumption: the network is a displayed tree of a reticulation graph extension"""
        for v_id in list(self.G.nodes):
            if not self.G.has_edge(v_id,f"{v_id}+"):
                continue
            #self._finalize_reticulation_graph(v_id,self.G.nodes[v_id]['ret_parent_match'])

            child_v_id = next((w_id for w_id in self.G.successors(v_id) if w_id != f"{v_id}+"), None)
            if child_v_id == None:
                continue
            child_v_id_plus = next((w_id for w_id in self.G.successors(f"{v_id}+") if str(w_id).endswith('+')), None)
            if child_v_id_plus == None:
                continue
            dprint(f"consider edges ({v_id},{child_v_id}) and ({v_id}+,{child_v_id_plus})")
            d_H11 = int(min(self.G.nodes[v_id]['stateset']) != min(self.G.nodes[child_v_id]['stateset']))
            d_H12 = int(min(self.G.nodes[self.G.nodes[v_id]['ret_parent_match']]['stateset']) != min(self.G.nodes[child_v_id]['stateset']))
            d_H21 = int(min(self.G.nodes[f"{v_id}+"]['stateset']) != min(self.G.nodes[child_v_id_plus]['stateset']))
            d_H22 = int(min(self.G.nodes[self.G.nodes[f"{v_id}+"]['ret_parent_match']]['stateset']) != min(self.G.nodes[child_v_id_plus]['stateset']))    
            if self.G.nodes[child_v_id] == 'reticulation' and self.G.nodes[child_v_id_plus] == 'reticulation':
                # v_id induces a fully unresolved reticulation graph
                if d_H11+d_H22 <= d_H12+d_H21:
                    # delete reticulation edge (v_id,child_v_id)
                    self._delete_ret_edge(self.G.nodes[v_id]['ret_parent_match'], v_id, child_v_id, self.get_sibling(v_id, child_v_id), next(self.G.successors(child_v_id),None), True)
                    # delete reticulation edge (v_id+,child_v_id_plus)
                    self._delete_ret_edge(self.G.nodes[f"{v_id}+"]['ret_parent_match'], f"{v_id}+", child_v_id_plus, self.get_sibling(f"{v_id}+", child_v_id_plus), next(self.G.successors(child_v_id_plus),None), True)
                else:
                    # delete reticulation edge (self.G.nodes[v_id]['ret_parent_match'],child_v_id)
                    self._delete_ret_edge(v_id, self.G.nodes[v_id]['ret_parent_match'], child_v_id, self.get_sibling(self.G.nodes[v_id]['ret_parent_match'], child_v_id), next(self.G.successors(child_v_id),None), True)
                    # delete reticulation edge (self.G.nodes[f"{v_id}+"]['ret_parent_match'],child_v_id_plus)
                    self._delete_ret_edge(f"{v_id}+", self.G.nodes[f"{v_id}+"]['ret_parent_match'], child_v_id_plus, self.get_sibling(self.G.nodes[f"{v_id}+"]['ret_parent_match'], child_v_id_plus), next(self.G.successors(child_v_id_plus),None), True)
            if self.G.nodes[child_v_id] == 'reticulation' and self.G.nodes[child_v_id_plus] != 'reticulation':
                    # delete reticulation edge (v_id,child_v_id)
                    self._delete_ret_edge(self.G.nodes[v_id]['ret_parent_match'], v_id, child_v_id, self.get_sibling(v_id, child_v_id), next(self.G.successors(child_v_id),None), True)
            if self.G.nodes[child_v_id] != 'reticulation' and self.G.nodes[child_v_id_plus] == 'reticulation':
                    # delete reticulation edge (self.G.nodes[f"{v_id}+"]['ret_parent_match'],child_v_id_plus)
                    self._delete_ret_edge(f"{v_id}+", self.G.nodes[f"{v_id}+"]['ret_parent_match'], child_v_id_plus, self.get_sibling(self.G.nodes[f"{v_id}+"]['ret_parent_match'], child_v_id_plus), next(self.G.successors(child_v_id_plus),None), True)
            if self.G.nodes[child_v_id] != 'reticulation' and self.G.nodes[child_v_id_plus] != 'reticulation':
                    dprint(f"The tree contains both ({v_id},{child_v_id}) and ({v_id}+,{child_v_id_plus})")
                    dprint(f"stateset of {v_id} = {self.G.nodes[v_id]['stateset']}, of {child_v_id} = {self.G.nodes[child_v_id]['stateset']}")
                    dprint(f"stateset of {self.G.nodes[v_id]['ret_parent_match']} = {self.G.nodes[self.G.nodes[v_id]['ret_parent_match']]['stateset']}, of {child_v_id} = {self.G.nodes[child_v_id]['stateset']}")
                    
                    if d_H11 > d_H21:
                        dprint(f"delete ({v_id},{child_v_id})")
                        self.G.remove_edge(v_id,child_v_id)
                        dprint(f"add ({self.G.nodes[v_id]['ret_parent_match']},{child_v_id})")
                        self.G.add_edge(self.G.nodes[v_id]['ret_parent_match'],child_v_id)
                    elif d_H11 < d_H21:
                        dprint(f"delete ({v_id}+,{child_v_id_plus})")
                        self.G.remove_edge(f"{v_id}+",child_v_id_plus)
                        p = self.G.nodes[f"{v_id}+"]['ret_parent_match']
                        dprint(f"add ({p},{child_v_id_plus})")
                        self.G.add_edge(self.G.nodes[f"{v_id}+"]['ret_parent_match'],child_v_id_plus)

    def _recover_final_tree(self):
        dprint("Recover final tree:")
        dprint("Reverse the reticulation graph extension")
        for v_id in list(self.G.nodes):
            if not self.G.has_node(f"{v_id}-"):# or self.G.has_edge(v_id,f"{v_id}+"):
                continue
            dprint(f"{v_id} has a non-adjacent duplicate {v_id}-")
            # v_id is the child of an original reticulation
            self.G.nodes[v_id]['stateset'] |= self.G.nodes[f"{v_id}-"]['stateset']
            for u_id in list(self.G.predecessors(f"{v_id}-")):
                dprint(f"add ({u_id},{v_id}) for parent {u_id} of {v_id}-")
                self.G.add_edge(u_id,v_id)
            self.G.nodes[v_id]['type'] = 'reticulation'
            dprint(f"delete {v_id}- and attach initial children to {v_id}")
            self.delete_vertex(f"{v_id}-")
            self.add_vertex(f"{v_id}c",'tree')
            self.G.add_edge(v_id,f"{v_id}c")
            self.G.nodes[f"{v_id}c"]['stateset'] = self.G.nodes[v_id]['stateset']
            for w_id in self.G.nodes[v_id]['initial_children']:
                self.G.add_edge(f"{v_id}c",w_id)
            self.G.nodes[v_id]['initial_children'] = None
        # suppress edge subdivisions
        dprint("Remove edge subdivisions")
        for v_id in list(self.G.nodes):
            if self.G.in_degree(v_id) == 1 and self.G.out_degree(v_id) == 1:
                dprint(f"add ({next(self.G.predecessors(v_id))},{next(self.G.successors(v_id))}) to delete {v_id}")
                self.G.add_edge(next(self.G.predecessors(v_id)),next(self.G.successors(v_id)))
                self.delete_vertex(v_id)
            



