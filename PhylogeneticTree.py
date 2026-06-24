import random
import copy
import networkx as nx
from PhylogeneticNetwork import PhylogeneticNetwork
from collections import deque
from typing import Tuple, Optional, Set, Deque
from itertools import chain, combinations

UNPROCESSED = "unprocessed"
PROCESSED = "processed"
VERBOSE = False

def dprint(*args, **kwargs):
    if VERBOSE: print(*args, **kwargs)


class PhylogeneticTree(PhylogeneticNetwork):
    """
    Phylogenetic tree as a phylogenetic network with additional operations.
    """

    def __init__(self):
        super().__init__()

    def copy(self) -> "PhylogeneticTree":
        new_tree = PhylogeneticTree()
        new_tree.G = self.G.copy(as_view=False)
        for attr, value in self.__dict__.items():
            if attr != "G":
                setattr(new_tree, attr, copy.deepcopy(value))
        return new_tree

    def _validate_tree(self):
        for v_id, attrs in self.G.nodes(data=True):
            if attrs.get('type') == 'reticulation':
                raise ValueError(f"Invalid tree: found reticulation vertex {v_id}")
            #if attrs.get('type') == 'subdivision':
            #    raise ValueError(f"Invalid tree: found edge subdivision {v_id}")

    def _process_tree_vertex_set_of_statesets(self, v_id, l) -> bool:
        """Processes a tree or root vertex, updating its set of statesets based on children.
           Assumption: the 'stateset' attributes are given as lists of sets
           Note: the resolution process turns the 'stateset' attributes into sets
        """
        if self.G.nodes[v_id]['status'] != UNPROCESSED or self.G.nodes[v_id]['type'] not in ['tree', 'root', 'subdivision']:
            return False
        children = list(self.G.successors(v_id))
        if not children:
            if self.G.nodes[v_id]['type'] == 'root':
                self.G.nodes[v_id]['status'] = PROCESSED
                return True
            else:
                return False

        if len(children) != 2:
            if self.G.nodes[v_id]['type'] == 'subdivision':
                if self.G.nodes[children[0]]['status'] == UNPROCESSED:
                    return False
                self.G.nodes[v_id]['stateset'] = self.G.nodes[children[0]]['stateset']
                self.G.nodes[v_id]['status'] = PROCESSED
                return True
            print(f"Warning: Tree/Root vertex {v_id} has {len(children)} children.")
            return False

        c1, c2 = children
        if self.G.nodes[c1]['status'] == UNPROCESSED or self.G.nodes[c2]['status'] == UNPROCESSED:
            return False

        def valid_subsets(max_set,min_set):
            """Generate all sets which contain min_set and are contained in max_set"""
            return (set(subset) for r in range(len(list(max_set))+1) 
                                for subset in combinations(list(max_set), r)
                                if min_set <= set(subset))

        # propagate sets of statesets
        lambda1 = self.G.nodes[c1]['stateset']
        lambda2 = self.G.nodes[c2]['stateset']
        intersection = [s for s in lambda1 if s in lambda2]
        if intersection:
            self.G.nodes[v_id]['stateset'] = [intersection[0]]
        elif len(lambda1) == 1 and len(lambda2) == 1 and len(lambda1[0]|lambda2[0])<=l:
            self.G.nodes[v_id]['stateset'] = list([lambda1[0]|lambda2[0]])
        else:
            for s1 in lambda1:
                for  s2 in lambda2:
                    for subset in valid_subsets(s1|s2,s1&s2):
                        if len(subset)>=1 and len(subset)==l:
                            self.G.nodes[v_id]['stateset'].append(subset)
        # resolve sets of statesets
        if self.G.nodes[v_id]['type'] == 'root':
            # degree of freedom: the assigned stateset can be chosen randomly
            self.G.nodes[v_id]['stateset'] = self.G.nodes[v_id]['stateset'][0]
        if isinstance(self.G.nodes[v_id]['stateset'], set):
            self._resolve_set(v_id)
        self.G.nodes[v_id]['status'] = PROCESSED
        return True

    def _resolve_set(self, v_id) -> None:
        """Propagates a character stateset assignment downwards from a resolved vertex."""
        stack = [v_id]
        while stack:
            vertex = stack.pop()
            for child in reversed(list(self.G.successors(vertex))):
                self.G.nodes[child]['stateset'] = max(self.G.nodes[child]['stateset'], key=lambda s: len(s & self.G.nodes[vertex]['stateset']))
                stack.append(child)

    def _iterative_processing_tree(self, l, process_subdivision=False) -> None:
        queue:       deque = deque()
        id_queue:    set[int] = set()

        def _add_to_queue(V,addChildren=False):
            for v in V:
                if self.G.nodes[v]['type'] in ('tree', 'root') or (process_subdivision and self.G.nodes[v]['type']=='subdivision'):
                    if self.G.nodes[v]['status'] != UNPROCESSED or v in id_queue:
                        continue
                    if (self.G.nodes[v]['type'] == 'root' and self.G.out_degree(v)==0) or (self.G.out_degree(v)>=1 and all(self.G.nodes[c]['status'] == PROCESSED for c in self.G.successors(v))):
                        queue.append(v); id_queue.add(v)
                elif (self.G.nodes[v]['type'] != 'leaf' and not process_subdivision) or (not self.G.nodes[v]['type'] in ('leaf','subdivision')):
                    print(f"Warning: vertex ",v," has type ",self.G.nodes[v]['type']," and is not added.")

        _add_to_queue(list(self.G.nodes))

        while queue:
            v = queue.popleft(); id_queue.discard(v)
            if self._process_tree_vertex_set_of_statesets(v,l):
                _add_to_queue(list(self.G.predecessors(v)),True)

        if all(self.G.nodes[v]['status'] == PROCESSED for v in self.G.nodes):
            return






