class Graph:
    def __init__(self):
        self.edges = []
        self.vertices = set()
        
    def add_edge(self, v, w, weight):
        self.vertices.add(v)
        self.vertices.add(w)
        self.edges.append((v,w,weight))
        
    def E(self):
        return self.edges
     
    def V(self):
        return self.vertices


class UnionFind:
    def __init__(self, vertices):
        self.size = len(vertices)
        self.components = {}
        self.tree_sizes = {}
        for v in vertices:
            self.components[v] = v
            self.tree_sizes[v] = 1
        
    def connect(self, a, b):
        a_root = self.root(a)
        b_root = self.root(b)
        if(a_root is not b_root): 
            if(self.tree_sizes[a_root] < self.tree_sizes[b_root]):
                self.components[a_root] = b_root
                self.tree_sizes[b_root] += self.tree_sizes[a_root]
            else: 
                self.components[b_root] = a_root
                self.tree_sizes[a_root] += self.tree_sizes[b_root]
       
    def connected(self, a, b):
        return self.root(a) is self.root(b)
    
    def root(self, a):
        while a is not self.components[a]:
            # path compression optimization
            self.components[a] = self.components[self.components[a]] 
            a = self.components[a]
        return a

def byWeight(edge):
    return edge[2]

# kruskal's algorithm
def minimum_spanning_tree(tree):
    result = []
    uf = UnionFind(tree.V())
    edges = sorted(tree.E(), key=byWeight)
    while edges:
        minor_edge = edges[0]
        edges.remove(minor_edge)
        v = minor_edge[0]
        w = minor_edge[1]
        if not uf.connected(v, w):
            result.append(minor_edge)
            uf.connect(v, w)
    return result
