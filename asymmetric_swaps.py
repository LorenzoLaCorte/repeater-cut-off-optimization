import copy
import itertools
from typing import List, Optional, Tuple

from gp_utils import catalan_number

class EdgeTreeNode:
    """
    Definition of a binary tree node without edge weights
        used as a helper for storing edge values of a TreeNode
    """
    def __init__(self, val=0, left=None, right=None, left_weight=0, right_weight=0):
        self.val = val
        self.left = left
        self.right = right


    def get_node_values(self):
        """
        Perform a DFS traversal and returns a tuple of the node values.
        """
        values = []
        def dfs(node: EdgeTreeNode) -> None:
            if node is None:
                return ''
            dfs(node.left), dfs(node.right)
            values.append(node.val)
        dfs(self)
        return tuple(values)
        
    def __repr__(self) -> str:
        """
        Recursive representation of the tree for debugging purposes.
        """
        repr_str = f'EdgeNode(val={self.val}, '

        # Represent left child and right child
        if self.left:
            repr_str += f'left={self.left.val}, '
        else:
            repr_str += f'left=None, '

        if self.right:
            repr_str += f'right={self.right.val})\n'
        else:
            repr_str += f'right=None)\n'

        # Recursively print left and right children (if they exist)
        if self.left:
            repr_str += self.left.__repr__()
        if self.right:
            repr_str += self.right.__repr__()

        return repr_str
    

class TreeNode:
    """
    Definition of a binary tree node with edge weights
    """
    def __init__(self, val=0, left=None, right=None, left_weight=0, right_weight=0):
        self.val = val
        self.left = (left, left_weight)
        self.right = (right, right_weight)

    def label_nodes(self) -> None:
        """
        Adds labels to the tree through a DFT
        with a number of leaves L = (n+1)/2 where n is the number of nodes
            - leaves are enumerated incrementally from the leftmost (label '0,') to the rightmost (label f'{L-1},')
            - parents are enumerated with the concatentaion of left and right labels
            -> the root of the tree should have label f'0,1,...,{L-1},'
        """
        leaf_enumeration = 0

        def dfs(node: TreeNode) -> None:
            nonlocal leaf_enumeration
            if node is None:
                return ''
            
            dfs(node.left[0]), dfs(node.right[0])

            if node.left[0] is None and node.right[0] is None:
                node.val = f'{leaf_enumeration},'
                leaf_enumeration += 1
            else:
                node.val = f'{node.left[0].val}{node.right[0].val}'
        dfs(self)
    
    
    def get_sequence(self) -> Tuple[str]:
        """
        Convert a FBT with labels in nodes and edges to a sequence (x_1, x_2, ...)
            where x_i is:
                - f's{l}' if it comes from the visit of a node with label l
                - f'd{l}' if it comes from the visit of an edge with label l with l>0
                    .. if l==0, the element f'd{l}' is not added to the sequence
        
        Leaves do not assign any value to the sequence
            .. if the node does not have child, it is a leaf (skip)
        When visiting a node
            - first check for the edges to left and right children
                .. if the edge has a label l>0, add it to the sequence as f'd{l}'
            - then add the node label to the sequence as f's{l}'
        """
        sequence = []

        def dfs(node: TreeNode, sequence: List[str]) -> None:
            if node.left[0] is None and node.right[0] is None:
                return ''
        
            dfs(node.left[0], sequence)
            dfs(node.right[0], sequence)
            
            # Append weights number of times the segment distillation
            if node.left[1] > 0:
                for _ in range(node.left[1]):
                    sequence.append(f'd{node.left[0].val[0]}')
            if node.right[1] > 0:
                for _ in range(node.right[1]):
                    sequence.append(f"d{node.right[0].val[0]}")
            sequence.append(f's{node.val[0]}')

        dfs(self, sequence)
        return tuple(sequence)


    def __repr__(self, level=0, indent="   ") -> str:
        """
        Recursive representation of the tree for debugging purposes.
        """
        repr_str = indent * level + f'Node(val={self.val}, '

        # Represent left child and left weight
        if self.left[0]:
            repr_str += f'left={self.left[0].val}, left_weight={self.left[1]}, '
        else:
            repr_str += f'left=None, left_weight={self.left[1]}, '

        # Represent right child and right weight
        if self.right[0]:
            repr_str += f'right={self.right[0].val}, right_weight={self.right[1]})\n'
        else:
            repr_str += f'right=None, right_weight={self.right[1]})\n'

        # Recursively print left and right children (if they exist)
        if self.left[0]:
            repr_str += self.left[0].__repr__(level + 1, indent)
        if self.right[0]:
            repr_str += self.right[0].__repr__(level + 1, indent)

        return repr_str


def get_all_FBT_shapes(n: int) -> List[Optional[TreeNode]]:
        """
        Returns all possible binary trees with n nodes
            all binary trees are dummy, i.e., each node has label 0
        """
        if n % 2 == 0:
            return []
        
        dp = {0 : [], 1 : [TreeNode()]}

        def backtrack(n: int) -> List[TreeNode]:
            # Base Case
            if n in dp:
                return [copy.deepcopy(tree) for tree in dp[n]]

            # Inductive Case
            results: List[TreeNode] = []
            
            # Consider all the possible numbers of
            #   r nodes on the left (l nodes on the right)
            for l in range(n):
                r = n-l-1
                leftTrees, rightTrees = backtrack(l), backtrack(r)
                for (t1, t2) in itertools.product(leftTrees, rightTrees):
                    root = TreeNode(val=0, left=t1, right=t2)
                    results.append(root)
            
            dp[n] = results
            return results

        return backtrack(n)


def get_all_edges_combs(n: int, tree: TreeNode, max_dists: int):
    """
    Returns all the possible combinations of values for the edges of a tree
    From a tree shape with n labeled nodes and (n-1) edges, 
        it generates max_dists^(n-1) possible trees with different values for the labels 
    """
    def dfs(node: TreeNode, edge_values: List[int], current_index: int) -> List[TreeNode]:
        """
        Perform a DFS traversal of the tree and assign edge values from the provided list.
        """
        if node is None:
            return None

        # Set the weights of the current node's left and right edges
        if node.left[0] is not None:
            node.left = (node.left[0], edge_values[current_index])
            current_index += 1
        if node.right[0] is not None:
            node.right = (node.right[0], edge_values[current_index])
            current_index += 1
        
        # Recursively apply DFS to the left and right subtrees
        if node.left[0] is not None:
            dfs(node.left[0], edge_values, current_index)
        if node.right[0] is not None:
            dfs(node.right[0], edge_values, current_index)

        return node
    
    n_edges = n-1
    
    if True: # DEBUG
        all_edge_combinations = list(itertools.product(range(max_dists+1), repeat=n_edges))
    else:
        all_edge_combinations = get_tree_edge_combs(tree, max_dists, n_edges)
    
    # Generate a new tree for each edge value combination
    for edge_values in all_edge_combinations:
        labeled_tree = copy.deepcopy(tree)
        dfs(labeled_tree, edge_values=list(edge_values), current_index=0)
        yield labeled_tree


def get_tree_edge_combs(swap_tree: TreeNode, max_dists: int, swap_tree_edges: int):
    """
    Return all possible combinations of edge values for a tree
        by creating an edge tree with (n_edges+1) nodes where 
            each node is assigned to the edges values in a dfs traversal

    """
    edge_trees: List[EdgeTreeNode] = [EdgeTreeNode()]
    
    def get_tree_edge_combs_dfs(swap_tree: TreeNode, edge_trees: EdgeTreeNode, weight_remaining: int) -> EdgeTreeNode:
        """
        Perform a DFS traversal of both trees together 
            (the two trees will have the same shape in the end)
            and assign swap tree's edge values to the edge tree nodes in a DFS traversal
        """
        if swap_tree is None:
            return None

        # Navigate the swap tree and return possible edge values
        # Edge Trees already existing in the list can be combined
        # -- with left and right nodes possibly having a weight from 0 to weight_left
        if swap_tree.left[0] is not None:
            for node_weight in range(weight_remaining):
                pass
                # edge_tree.left = EdgeTreeNode(val=node_weight)
                # get_tree_edge_combs_dfs(swap_tree.left[0], edge_tree.left, weight_remaining-node_weight)
        
        if swap_tree.right[0] is not None:
            for node_weight in range(weight_remaining):
                pass
                # edge_tree.right = EdgeTreeNode(val=node_weight)
                # get_tree_edge_combs_dfs(swap_tree.right[0], edge_tree.right, weight_remaining-node_weight)

        return edge_trees

    yield from get_tree_edge_combs_dfs(swap_tree, edge_trees, max_dists)
    

def generate_swap_space(S: int):
    """
    Returns all possible sequences of swaps 
        built from binary trees where nodes are labeled.
    From the number of segments S (leaves in the tree)
        we can derive the number of nodes n = 2S-1
    """
    n: int = 2*S - 1
    shapes: List[TreeNode] = get_all_FBT_shapes(n)

    for shape in shapes:
        shape.label_nodes()
        yield shape, shape.get_sequence()


def generate_asym_protocol_space(N: int, max_dists: int):
    """
    Returns all possible sequences of swaps and distillations 
        built from binary trees where both nodes and edges are labeled.
    
    Each tree shape corresponds to a sequence of swaps.
    For each tree shape:
        - we can label all nodes 
        - we can generate all possible combinations of edge labels (from 0 to max_dists) for the single tree
        Then, for each of the label tree we can extract sequences
    """
    S = N - 1
    n = 2*S - 1
    for nodeLabeledShape, _ in generate_swap_space(S):
        for edgeLabeledShape in get_all_edges_combs(n, nodeLabeledShape, max_dists):
            yield edgeLabeledShape.get_sequence()


if __name__ == "__main__":
    N = 5
    S = N - 1
    max_dists = 2
    
    print(f"\nExpected cardinality of swap space for {N} nodes ({S} segments): {catalan_number(N-2)}")
    swap_space = list(generate_swap_space(S))
    print(f"   Total cardinality of swap space for {N} nodes ({S} segments): {len(swap_space)}")

    expected = catalan_number(N-2) * (max_dists+1)**((2*S-1)-1)
    print(f"\nExpected number of asymmetric protocols for {N} nodes ({S} segments) and max_dists={max_dists}: {expected}")
    asym_protocol_space = list(generate_asym_protocol_space(N, max_dists))
    print(f"   Total number of asymmetric protocols for {N} nodes ({S} segments) and max_dists={max_dists}: {len(asym_protocol_space)}")
    