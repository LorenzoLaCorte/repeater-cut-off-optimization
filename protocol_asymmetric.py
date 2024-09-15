import copy
import itertools
import logging
import math
import statistics
from typing import Generator, List, Optional, Tuple
from repeater_types import checkAsymProtocol


class SwapTreeVertex:
    """
    Definition of a binary tree vertex labeled with attributes for the segments and distillations.
    Attributes:
        - segments: A tuple (i, j) where
            - i is the leftmost link of the chain represented by the node
            - j is the rightmost link of the chain represented by the node
        - dists: The number of distillations performed on this segment
        - left: The left child of the current vertex in the binary tree 
        - right: The right child of the current vertex in the binary tree
    """
    def __init__(self, segments=(0, 0), dists=0, left=None, right=None):
        self._validate_segments(segments)
        self.segments: int = segments
        self.dists: int = dists
        self.left: SwapTreeVertex | None = left
        self.right: SwapTreeVertex | None = right
        self.visited: bool = False

    @staticmethod
    def _validate_segments(segments):
        if not isinstance(segments, tuple):
            raise ValueError("The segments attribute must be a tuple")
        if len(segments) != 2:
            raise ValueError("The segments attribute must be a tuple of length 2")

    def visit(self) -> None:
        """
        Marks the vertex as visited.
        """
        self.visited = True

    def all_visited(self) -> bool:
        """
        Returns True if all the vertices of the tree have been visited.
        """
        visited_statuses, stack = [], [self]
        while stack:
            vertex = stack.pop()
            visited_statuses.append(vertex.visited)
            if vertex.right:
                stack.append(vertex.right)
            if vertex.left:
                stack.append(vertex.left)

        return all(visited_statuses)

    def label_vertices(self) -> None:
        """
        Adds labels to the tree through a DFT
        with a number of leaves L = (v+1)/2 where v is the number of vertices
            - leaves are enumerated incrementally from the leftmost [segments (0-0)] to the rightmost [segments (L-1, L-1)]
            - parents are enumerated with the concatentaion of left and right labels
            -> the root of the tree should have segments (0, L-1)
        """
        leaf_enumeration = 0

        def dfs(node: SwapTreeVertex) -> None:
            nonlocal leaf_enumeration
            if node is None:
                return ''

            dfs(node.left), dfs(node.right)

            if node.left is None and node.right is None:
                node.segments = (leaf_enumeration, leaf_enumeration)
                leaf_enumeration += 1
            else:
                node.segments = (node.left.segments[0], node.right.segments[1])
        dfs(self)
    
    
    def get_sequence(self) -> Tuple[str]:
        """
        Convert a FBT with labels in vertices to a sequence (x_1, x_2, ...)
            where x_i is:
                - f's{l}' if it comes from the visit of a vertex with label l
                - f'd{l}' if it comes from the visit of an edge with label l with l>0
                    .. if l==0, the element f'd{l}' is not added to the sequence
        """
        sequence = []

        def dfs(node: SwapTreeVertex, sequence: List[str]) -> None:
            if node.left is None and node.right is None:
                return
            dfs(node.left, sequence)
            dfs(node.right, sequence)
            
            if node.left.dists > 0:
                for _ in range(node.left.dists):
                    sequence.append(f'd{node.left.segments[1]}')
            if node.right.dists > 0:
                for _ in range(node.right.dists):
                    sequence.append(f"d{node.right.segments[1]}")

            sequence.append(f's{node.left.segments[1]}')
        dfs(self, sequence)
        
        # Add the last distillation (not covered by the DFS)
        if self.dists > 0:
            for _ in range(self.dists):
                sequence.append(f'd{self.segments[1]}')
        
        return tuple(sequence)

    def get_leaf_depths(self, depth=0, leaf_depths=None):
        """
        Perform a depth-first search and extract all depths of the leaves in the binary tree.
        
        Args:
            self (SwapTreeVertex): The current vertex of the binary tree.
            depth (int): The current depth of the tree (default is 0).
            leaf_depths (list): A list that will store the depths of the leaves (default is None).
            
        Returns:
            list: A list of depths for all the leaves in the tree.
        """
        if leaf_depths is None:
            leaf_depths = []
        
        if self.left is None and self.right is None:
            leaf_depths.append(depth)
        else:
            if self.left is not None and self.right is not None:
                self.left.get_leaf_depths(depth + 1, leaf_depths)
                self.right.get_leaf_depths(depth + 1, leaf_depths)

        return leaf_depths


    def get_symmetry_score(self) -> float:
        """
        Returns a normalized score for the symmetry of the entire tree.
        The score is computed based on the variance of the leaf depths:
        - A perfect symmetry gives a score of 1.
        - Higher variance in leaf depths lowers the symmetry score.
        
        Returns:
            float: The symmetry score between 0 and 1.
        """
        leaf_depths = self.get_leaf_depths()
        if len(leaf_depths) <= 1:
            return 1.0  # trivially symmetric

        variance = statistics.variance(leaf_depths)
        
        # Max variance corresponds to a situation where leaf depths are spread out
        max_variance = max(leaf_depths)  
        symmetry_score = max(0, 1 - (variance / max_variance))  # Symmetry score ranges between 0 and 1
        return symmetry_score
        

    def count_nodes(self) -> int:
        """
        Returns the number of nodes in the subtree rooted at this node.
        """
        if not self:
            return 0
        left_count = self.left.count_nodes() if self.left else 0
        right_count = self.right.count_nodes() if self.right else 0
        return 1 + left_count + right_count

    def __repr__(self, level=0, indent="   ") -> str:
        """
        Recursive representation of the tree for debugging purposes.
        """
        ret = f"{indent * level}STN({self.segments[0]}-{self.segments[1]}, {self.dists})\n"
        if self.left:
            ret += f"{self.left.__repr__(level + 1, indent)}\n"
        if self.right:
            ret += f"{self.right.__repr__(level + 1, indent)}\n"
        return ret.rstrip()

def get_all_FBT_shapes(vertices: int) -> List[Optional[SwapTreeVertex]]:
    """
    Returns all possible binary trees with v vertices
        all binary trees are dummy and represent tree shapes, 
            each node has no label (default segments=(0, 0), dists=0)
    """
    if vertices % 2 == 0:
        return []
    
    dp = {0 : [], 1 : [SwapTreeVertex()]}

    def backtrack(v: int) -> List[SwapTreeVertex]:
        if v in dp:
            return [copy.deepcopy(tree) for tree in dp[v]]

        # Consider all the possible numbers of nodes in the subtrees
        results: List[SwapTreeVertex] = []
        for l in range(v):
            r = v-l-1
            leftTrees, rightTrees = backtrack(l), backtrack(r)
            for (t1, t2) in itertools.product(leftTrees, rightTrees):
                root = SwapTreeVertex(left=t1, right=t2)
                results.append(root)
        
        dp[v] = results
        return results

    return backtrack(vertices)


def assign_dists_to_tree(tree: SwapTreeVertex, dists_comb: List[int]) -> SwapTreeVertex:
    """
    Assigns distillation values to the vertices of a tree.
    """
    # logging.info(f"Assigning distillation values {dists_comb} to the tree {tree}")
    def dfs(tree: SwapTreeVertex, dists_values: List[int], current_index: int) -> Tuple[SwapTreeVertex, int]:
        """
        Perform a DFS traversal of the tree and assign distillation values from the provided list.
        Assumes the tree is a full binary tree.
        First values of the dists_values list are assigned to the leaves of the tree.
        Then, leaves are removed from the list and the process is repeated for the subtrees.
        """
        if tree is None:
            return None, current_index

        # If the current node is a leaf or both children have been visited, assign the distillation values
        if (tree.left is None and tree.right is None) or (tree.left.visited and tree.right.visited):
            if not tree.visited: 
                tree.visit()
                tree.dists = dists_values[current_index]  # TODO: Maybe is better [current_index-1]
                current_index += 1
        else:
            # Recursively apply DFS to the left and right subtrees
            tree.left, current_index = dfs(tree.left, dists_values, current_index)
            tree.right, current_index = dfs(tree.right, dists_values, current_index)

        return tree, current_index

    dists_comb_tree = copy.deepcopy(tree)
    idx = 0

    # Until all the vertices have been visited, assign distillation values to the tree
    while not dists_comb_tree.all_visited():
        dists_comb_tree, idx = dfs(dists_comb_tree, dists_values=list(dists_comb), current_index=idx)

    return dists_comb_tree


def generate_dists_combs(vertices: int, tree: SwapTreeVertex, max_dists: int) -> Generator[SwapTreeVertex, None, None]:
    """
    Returns all the possible combinations of values for the distillations of a tree
    From a tree shape with v labeled vertices, 
        it generates (max_dists+1)^(v) possible trees with different values for the labels 
    """
    if vertices == 1:
        for i in range(max_dists+1):
            yield SwapTreeVertex(segments=(0, 0), dists=i)
        return

    # Generate a new tree for each distillation values combination
    for dists_comb in itertools.product(range(max_dists+1), repeat=vertices):
        yield assign_dists_to_tree(tree, dists_comb)


def generate_swap_space(S: int):
    """
    Returns all possible sequences of swaps
        built from binary trees where nodes are labeled.
    From the number of segments S (leaves in the tree)
        we can derive the number of vertices = 2S-1
    """
    v: int = 2*S - 1
    shapes: List[SwapTreeVertex] = get_all_FBT_shapes(v)

    for shape in shapes:
        shape.label_vertices()
        yield shape, shape.get_sequence()


def generate_asym_protocol_space(N: int, max_dists: int):
    """
    Returns all possible sequences of swaps and distillations 
        built from binary trees where nodes are labeled with segments and distillations.
    
    Each tree shape corresponds to a sequence of swaps.
    For each tree shape:
        - we can label all nodes
        - we can generate all possible combinations of dists labels (from 0 to max_dists) for the single tree
        Then, for each of the label tree we can extract sequences
    """
    S = N - 1
    v = 2*S - 1
    for nodeLabeledShape, _ in generate_swap_space(S):
        for edgeLabeledShape in generate_dists_combs(v, nodeLabeledShape, max_dists):
            yield edgeLabeledShape.get_sequence()