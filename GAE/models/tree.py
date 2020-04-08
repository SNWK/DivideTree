
import sys
sys.path.insert(0, '../helpers')
sys.path.insert(0, '../../')
from utils.divtree_gen import *
from datautils import *


class TreeNode(object):

    def __init__(self, feature, idx):
        self.idx = idx
        self.feature = feature
        self.neighbors = []
        self.label = self.feature
        
    def add_neighbor(self, nei_node):
        self.neighbors.append(nei_node)
    

class DTree(object):

    def __init__(self, peaks):
        
        self.nodes = dict()

        # clique contains componetes , len = 1 -> singleton, len > 1 ring ...
        # edges: the tree's connections over cliques

        self.root, edges, self.peaks = tree_extract(peaks)

        for i,p in enumerate(peaks.index):
            node = TreeNode(get_feature(p, peaks), i)
            self.nodes[p] = node

        for x,y in edges:
            self.nodes[x].add_neighbor(self.nodes[y])
            self.nodes[y].add_neighbor(self.nodes[x])

    def size(self):
        return len(self.nodes)

def dfs(node, fa_idx):
    max_depth = 0
    for child in node.neighbors:
        if child.idx == fa_idx: continue
        max_depth = max(max_depth, dfs(child, node.idx))
    return max_depth + 1


if __name__ == "__main__":
    import sys
